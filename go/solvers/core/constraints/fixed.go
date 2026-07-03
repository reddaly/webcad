package constraints

import (
	"fmt"
	"math"

	"github.com/gonzojive/webcad/schema"
	"gonum.org/v1/gonum/mat"
)

type FixedEvaluator struct {
	entityID      string
	initialValues []float64
}

func NewFixedEvaluator(c *schema.Constraint, entities map[string]*schema.Entity) (*FixedEvaluator, error) {
	fixed := c.GetFixed()
	entID := fixed.GetEntityId()
	ent, ok := entities[entID]
	if !ok {
		return nil, fmt.Errorf("entity %s not found", entID)
	}
	params := getParams(ent)
	if params == nil {
		return nil, fmt.Errorf("failed to get params for entity %s", entID)
	}
	// Copy params to avoid sharing mutable state
	initialValues := make([]float64, len(params))
	copy(initialValues, params)

	return &FixedEvaluator{
		entityID:      entID,
		initialValues: initialValues,
	}, nil
}

func (f *FixedEvaluator) Evaluate(x []float64, grad []float64, paramIndices map[string]int) float64 {
	idx, ok := paramIndices[f.entityID]
	if !ok {
		return 0.0
	}
	count := len(f.initialValues)
	const weightSq = 1000.0
	totalResidualSq := 0.0

	for i := 0; i < count; i++ {
		diff := x[idx+i] - f.initialValues[i]
		totalResidualSq += weightSq * diff * diff
		if grad != nil {
			grad[idx+i] += 2.0 * weightSq * diff
		}
	}
	return totalResidualSq
}

func (f *FixedEvaluator) NumEquations() int {
	return len(f.initialValues)
}

func (f *FixedEvaluator) EvaluateJacobian(x []float64, residuals []float64, J *mat.Dense, rowOffset int, paramIndices map[string]int) {
	idx, ok := paramIndices[f.entityID]
	if !ok {
		return
	}
	count := len(f.initialValues)
	const weightSq = 1000.0
	sqrtW := math.Sqrt(weightSq)

	for i := 0; i < count; i++ {
		diff := x[idx+i] - f.initialValues[i]
		residuals[i] = sqrtW * diff
		if J != nil {
			J.Set(rowOffset+i, idx+i, sqrtW)
		}
	}
}
