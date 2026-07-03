// Package bfgs implements a webcad solver using the Gonum optimize package.
// It translates geometric entities and constraints into a continuous
// optimization problem and minimizes the sum of squared constraint residuals.
package bfgs

import (
	"errors"
	"fmt"
	"strings"
	"time"

	"github.com/gonzojive/webcad/go/solvers/core"
	"github.com/gonzojive/webcad/schema"

	"gonum.org/v1/gonum/optimize"
	"google.golang.org/protobuf/proto"
)

// Solver implements core.Solver using gonum/optimize.
type Solver struct {
	NumericalGradients bool
}

// New returns a new Gonum-based solver with analytical gradients by default.
func New() *Solver {
	return &Solver{NumericalGradients: false}
}

// NewNumerical returns a new Gonum-based solver using numerical gradients (finite differences).
func NewNumerical() *Solver {
	return &Solver{NumericalGradients: true}
}

// ID returns the unique identifier of this solver.
func (s *Solver) ID() core.SolverID {
	if s.NumericalGradients {
		return "bfgs_numerical"
	}
	return "bfgs_analytical"
}

// SolveCold solves the sketch from scratch.
// It implements a multi-trial loop to get accurate sub-millisecond measurements.
func (s *Solver) SolveCold(sketch *schema.Sketch) (*schema.SolveResult, error) {
	trials := sketch.GetColdRepeatedTrials()
	if trials <= 0 {
		trials = 100
	}

	// Save initial state to restore before each trial
	initialState := make(map[string][]float64)
	for _, ent := range sketch.Entities {
		initialState[ent.Id] = append([]float64(nil), core.GetParams(ent)...)
	}

	var err error
	var optResult *optimize.Result
	start := time.Now()
	for i := int32(0); i < trials; i++ {
		// Restore initial state
		for _, ent := range sketch.Entities {
			core.SetParams(ent, initialState[ent.Id])
		}
		optResult, err = s.solve(sketch)
		if err != nil {
			break
		}
	}
	duration := time.Since(start)

	if err != nil {
		return &schema.SolveResult{
			SketchId:  sketch.Id,
			SolverName:   string(s.ID()),
			Success:      false,
			ErrorMessage: err.Error(),
		}, nil
	}

	coldSolveTimeMs := (float64(duration.Nanoseconds()) / 1e6) / float64(trials)

	result := &schema.SolveResult{
		SketchId:     sketch.Id,
		SolverName:      string(s.ID()),
		ColdSolveTimeMs: coldSolveTimeMs,
		Success:         true,
	}
	if optResult != nil {
		result.Telemetry = &schema.SolverTelemetry{
			Iterations:      int32(optResult.Stats.MajorIterations),
			FuncEvaluations: int32(optResult.Stats.FuncEvaluations),
			GradEvaluations: int32(optResult.Stats.GradEvaluations),
			FinalResidual:   optResult.Location.F,
		}
	}
	s.populateSolvedState(result, sketch)
	return result, nil
}

// SolveWarm solves the sketch, simulating warm start.
// For this solver, it performs the same optimization as SolveCold but without the loop.
func (s *Solver) SolveWarm(sketch *schema.Sketch) (*schema.SolveResult, error) {
	start := time.Now()
	optResult, err := s.solve(sketch)
	duration := time.Since(start)

	result := &schema.SolveResult{
		SketchId:        sketch.Id,
		SolverName:         string(s.ID()),
		WarmSolveAvgTimeMs: float64(duration.Nanoseconds()) / 1e6,
		Success:            err == nil,
	}
	if err != nil {
		result.ErrorMessage = err.Error()
	} else if optResult != nil {
		result.Telemetry = &schema.SolverTelemetry{
			Iterations:      int32(optResult.Stats.MajorIterations),
			FuncEvaluations: int32(optResult.Stats.FuncEvaluations),
			GradEvaluations: int32(optResult.Stats.GradEvaluations),
			FinalResidual:   optResult.Location.F,
		}
	}

	s.populateSolvedState(result, sketch)
	return result, nil
}

// solve performs the common setup and execution of the optimization problem.
func (s *Solver) solve(sketch *schema.Sketch) (*optimize.Result, error) {
	// Initialize the new ConstraintSystem
	sys, err := core.NewConstraintSystem(sketch)
	if err != nil {
		return nil, err
	}
	initialX := sys.ExtractVariables()

	if len(initialX) == 0 {
		return nil, errors.New("no optimizable entities found")
	}

	// Check if the initial state is already solved.
	initialObj := sys.Objective(initialX)
	if initialObj < 1e-12 {
		sys.UpdateSketch(initialX)
		return &optimize.Result{
			Location: optimize.Location{F: initialObj},
		}, nil
	}

	// Configure the problem with analytical or numerical Grad.
	problem := optimize.Problem{
		Func: sys.Objective,
	}
	if s.NumericalGradients {
		problem.Grad = func(grad, x []float64) {
			h := 1e-6
			for i := range x {
				temp := x[i]
				x[i] = temp - h
				fMinus := sys.Objective(x)
				x[i] = temp + h
				fPlus := sys.Objective(x)
				x[i] = temp
				grad[i] = (fPlus - fMinus) / (2.0 * h)
			}
		}
	} else {
		problem.Grad = sys.ObjectiveGradient
	}

	// Configure settings to ensure we converge to a very high precision.
	settings := &optimize.Settings{
		InitValues: &optimize.Location{
			F: initialObj,
		},
		Converger: &optimize.FunctionConverge{
			Absolute:   1e-12, // Tighter than default 1e-10
			Iterations: 200,   // Allow more iterations to find the true minimum
		},
	}

	// Use BFGS (quasi-Newton) for fast, superlinear local convergence.
	method := &optimize.BFGS{}

	result, err := runOptimization(problem, initialX, settings, method)
	if err != nil {
		return nil, err
	}

	sys.UpdateSketch(result.X)
	return result, nil
}

// runOptimization executes the minimization process.
func runOptimization(problem optimize.Problem, initialX []float64, settings *optimize.Settings, method optimize.Method) (*optimize.Result, error) {
	result, err := optimize.Minimize(problem, initialX, settings, method)
	if err != nil {
		// If it's a linesearch failure, we might still have a good enough result.
		if result != nil && strings.Contains(err.Error(), "linesearch") {
			if result.F < 1e-8 {
				return result, nil
			}
		}
		return nil, fmt.Errorf("optimization failed: %w", err)
	}

	if err := result.Status.Err(); err != nil {
		// Tolerate linesearch failure if the residual is already acceptable.
		if strings.Contains(err.Error(), "linesearch") {
			if result.F < 1e-8 {
				return result, nil
			}
		}
		return nil, fmt.Errorf("optimization did not converge: %w", err)
	}
	return result, nil
}

func (s *Solver) populateSolvedState(result *schema.SolveResult, sketch *schema.Sketch) {
	if !result.Success {
		return
	}
	result.SolvedState = &schema.StateSnapshot{
		Entities: make(map[string]*schema.Entity),
	}
	for _, ent := range sketch.Entities {
		result.SolvedState.Entities[ent.Id] = proto.Clone(ent).(*schema.Entity)
	}
}
