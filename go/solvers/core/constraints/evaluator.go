package constraints

import (
	"gonum.org/v1/gonum/mat"
)

// Evaluator defines the interface for a decomposed constraint evaluator.
type Evaluator interface {
	// Evaluate computes the squared residual of the constraint.
	// If grad is not nil, it also accumulates the gradient of the squared residual
	// into the global grad slice.
	//
	// x: the flat parameter vector.
	// grad: the global gradient vector to accumulate into (may be nil).
	// paramIndices: maps entity ID to its starting index in x.
	Evaluate(x []float64, grad []float64, paramIndices map[string]int) float64
}

// JacobianEvaluator allows second-order solvers (like LM) to extract
// individual residuals and Jacobian rows directly.
type JacobianEvaluator interface {
	Evaluator
	// NumEquations returns the number of independent scalar equations
	// this constraint represents.
	NumEquations() int
	// EvaluateJacobian evaluates the unsquared residuals and writes their
	// gradients (Jacobian rows) directly into the J matrix starting at rowOffset.
	// If J is nil, only the residuals are evaluated.
	EvaluateJacobian(x []float64, residuals []float64, J *mat.Dense, rowOffset int, paramIndices map[string]int)
}

