// Package core defines the primary interfaces and types used across the webcad
// solver library.
package core

import (
	"github.com/gonzojive/webcad/schema"
)

// Solver defines the interface that any constraint solver must implement.
//
// Implementations may delegate to external processes, use CGO wrapper,
// or be written purely in Go.
type Solver interface {
	// ID returns the unique identifier for this solver.
	// This ID should be consistent across runs and uniquely identify the
	// solver implementation (e.g., "lm", "bfgs").
	ID() SolverID

	// SolveCold takes a fresh sketch and attempts to solve it from scratch.
	// It should not rely on any previous solver state. The input sketch
	// may contain initial guesses for parameters, but the solver should
	// treat this as a cold start.
	SolveCold(sketch *schema.Sketch) (*schema.SolveResult, error)

	// SolveWarm takes a sketch that has already been partially solved or
	// dragged, simulating interactive use.
	//
	// This method is intended to solve the sketch during interactive editing,
	// where the input sketch is close to a solved state.
	// If a solver does not support warm starts, it should return a result
	// indicating failure or fallback to a cold solve, as appropriate.
	SolveWarm(sketch *schema.Sketch) (*schema.SolveResult, error)
}

