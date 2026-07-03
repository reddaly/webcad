// Package lm implements a 2D geometric constraint solver using the
// Levenberg-Marquardt (LM) optimization algorithm.
//
// It is optimized for high performance and zero heap allocations in the hot loop.
package lm

import (
	"errors"
	"fmt"
	"sync"
	"time"

	"github.com/gonzojive/webcad/go/solvers/core"
	"github.com/gonzojive/webcad/schema"
	"google.golang.org/protobuf/proto"
)

type solveResult struct {
	status          SolverStatus
	iterations      int
	funcEvaluations int
	gradEvaluations int
	finalResidual   float64
}

// LMSolver implements core.Solver using the Levenberg-Marquardt algorithm.
//
// It uses pre-allocated workspaces and direct BLAS/LAPACK calls to achieve
// zero heap allocations during solving.
type LMSolver struct {
	// EpGeom is the geometric tolerance for convergence.
	// The solver succeeds if the infinity norm of residuals is below this value.
	EpGeom  float64
	// EpGrad is the gradient tolerance.
	// The solver stops with Inconsistent status if the infinity norm of the gradient
	// is below this value, indicating a local minimum.
	EpGrad  float64
	// EpStep is the step tolerance.
	// The solver stops with Stalled status if the step size is below this value.
	EpStep  float64
	// MaxIter is the maximum number of iterations allowed.
	MaxIter int
	pool    sync.Pool
}

// New returns a new LMSolver with default tolerances and iteration limits.
func New() *LMSolver {
	return &LMSolver{
		EpGeom:  1e-8,
		EpGrad:  1e-12,
		EpStep:  1e-12,
		MaxIter: 100,
		pool: sync.Pool{
			New: func() interface{} {
				return &SolverWorkspace{}
			},
		},
	}
}

// ID returns the unique identifier "lm" for this solver.
func (s *LMSolver) ID() core.SolverID {
	return "lm"
}

// SolveCold solves the given sketch from scratch.
//
// It restores the initial state before solving and runs multiple trials
// if configured in the sketch to obtain average execution time.
// Returns a SolveResult containing the solved state and telemetry.
func (s *LMSolver) SolveCold(sketch *schema.Sketch) (*schema.SolveResult, error) {
	trials := sketch.GetColdRepeatedTrials()
	if trials <= 0 {
		trials = 100
	}

	// Save initial state to restore before each trial
	initialState := make(map[string][]float64)
	for _, ent := range sketch.Entities {
		initialState[ent.Id] = append([]float64(nil), core.GetParams(ent)...)
	}

	var res solveResult
	start := time.Now()
	for i := 0; i < int(trials); i++ {
		// Restore initial state
		for _, ent := range sketch.Entities {
			core.SetParams(ent, initialState[ent.Id])
		}

		sys, err := core.NewConstraintSystem(sketch)
		if err != nil {
			return nil, err
		}
		x := sys.ExtractVariables()
		if len(x) == 0 {
			return nil, errors.New("no optimizable entities found")
		}

		// Check if the initial state is already solved using configurable tolerance.
		initialObj := sys.Objective(x)
		if initialObj < s.EpGeom*s.EpGeom {
			sys.UpdateSketch(x)
			res = solveResult{
				status:        Success,
				finalResidual: initialObj,
			}
			continue
		}

		var xOpt []float64
		xOpt, res = s.solve(sys, x)
		sys.UpdateSketch(xOpt)
	}
	duration := time.Since(start)

	coldSolveTimeMs := (float64(duration.Nanoseconds()) / 1e6) / float64(trials)

	result := &schema.SolveResult{
		SketchId:     sketch.Id,
		SolverName:      string(s.ID()),
		ColdSolveTimeMs: coldSolveTimeMs,
		Success:         res.status == Success,
	}

	if !result.Success {
		result.ErrorMessage = fmt.Sprintf("solver failed with status: %v, final residual: %e", res.status, res.finalResidual)
	}

	result.Telemetry = &schema.SolverTelemetry{
		Iterations:      int32(res.iterations),
		FuncEvaluations: int32(res.funcEvaluations),
		GradEvaluations: int32(res.gradEvaluations),
		FinalResidual:   res.finalResidual,
	}

	s.populateSolvedState(result, sketch)
	return result, nil
}

// SolveWarm solves the given sketch, simulating a warm start.
//
// It assumes the sketch is already close to a solved state (e.g. during dragging).
// It runs a single solve trial and returns the result.
func (s *LMSolver) SolveWarm(sketch *schema.Sketch) (*schema.SolveResult, error) {
	sys, err := core.NewConstraintSystem(sketch)
	if err != nil {
		return nil, err
	}
	x := sys.ExtractVariables()
	if len(x) == 0 {
		return nil, errors.New("no optimizable entities found")
	}

	// Check if the initial state is already solved using configurable tolerance.
	initialObj := sys.Objective(x)
	if initialObj < s.EpGeom*s.EpGeom {
		sys.UpdateSketch(x)
		result := &schema.SolveResult{
			SketchId: sketch.Id,
			SolverName:  string(s.ID()),
			Success:     true,
		}
		result.Telemetry = &schema.SolverTelemetry{
			FinalResidual: initialObj,
		}
		s.populateSolvedState(result, sketch)
		return result, nil
	}

	start := time.Now()
	xOpt, res := s.solve(sys, x)
	duration := time.Since(start)

	sys.UpdateSketch(xOpt)

	result := &schema.SolveResult{
		SketchId:        sketch.Id,
		SolverName:         string(s.ID()),
		WarmSolveAvgTimeMs: float64(duration.Nanoseconds()) / 1e6,
		Success:            res.status == Success,
	}

	if !result.Success {
		result.ErrorMessage = fmt.Sprintf("solver failed with status: %v, final residual: %e", res.status, res.finalResidual)
	}

	result.Telemetry = &schema.SolverTelemetry{
		Iterations:      int32(res.iterations),
		FuncEvaluations: int32(res.funcEvaluations),
		GradEvaluations: int32(res.gradEvaluations),
		FinalResidual:   res.finalResidual,
	}

	s.populateSolvedState(result, sketch)
	return result, nil
}

func (s *LMSolver) populateSolvedState(result *schema.SolveResult, sketch *schema.Sketch) {
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
