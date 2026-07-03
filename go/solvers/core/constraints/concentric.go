package constraints

import (
	"fmt"
	"github.com/gonzojive/webcad/schema"
	"gonum.org/v1/gonum/mat"
)

type ConcentricEvaluator struct {
	idA, idB string
}

func NewConcentricEvaluator(c *schema.Constraint, entities map[string]*schema.Entity) (*ConcentricEvaluator, error) {
	cc := c.GetConcentric()
	idA := cc.GetEntityA()
	idB := cc.GetEntityB()
	if idA == idB {
		return nil, fmt.Errorf("concentric constraint cannot be applied to the same entity %s", idA)
	}
	entA, okA := entities[idA]
	if !okA {
		return nil, fmt.Errorf("entity A %s not found", idA)
	}
	if !isPointOrCenter(entA) {
		return nil, fmt.Errorf("entity A %s is not a point, circle, arc, or ellipse", idA)
	}
	entB, okB := entities[idB]
	if !okB {
		return nil, fmt.Errorf("entity B %s not found", idB)
	}
	if !isPointOrCenter(entB) {
		return nil, fmt.Errorf("entity B %s is not a point, circle, arc, or ellipse", idB)
	}
	return &ConcentricEvaluator{
		idA: idA,
		idB: idB,
	}, nil
}

func (c *ConcentricEvaluator) Evaluate(x []float64, grad []float64, paramIndices map[string]int) float64 {
	idx1, ok1 := paramIndices[c.idA]
	idx2, ok2 := paramIndices[c.idB]
	if !ok1 || !ok2 {
		return 0.0
	}

	// Centers are at x[idx] and x[idx+1] for Point, Circle, Arc, and Ellipse
	dx := x[idx1] - x[idx2]
	dy := x[idx1+1] - x[idx2+1]
	totalResidualSq := dx*dx + dy*dy

	if grad != nil {
		grad[idx1] += 2.0 * dx
		grad[idx1+1] += 2.0 * dy
		grad[idx2] -= 2.0 * dx
		grad[idx2+1] -= 2.0 * dy
	}
	return totalResidualSq
}

func (c *ConcentricEvaluator) NumEquations() int {
	return 2
}

func (c *ConcentricEvaluator) EvaluateJacobian(x []float64, residuals []float64, J *mat.Dense, rowOffset int, paramIndices map[string]int) {
	idx1, ok1 := paramIndices[c.idA]
	idx2, ok2 := paramIndices[c.idB]
	if !ok1 || !ok2 {
		return
	}

	dx := x[idx1] - x[idx2]
	dy := x[idx1+1] - x[idx2+1]

	residuals[0] = dx
	residuals[1] = dy

	if J != nil {
		// Row 0: r_0 = x_1 - x_2
		J.Set(rowOffset, idx1, 1.0)
		J.Set(rowOffset, idx2, -1.0)

		// Row 1: r_1 = y_1 - y_2
		J.Set(rowOffset+1, idx1+1, 1.0)
		J.Set(rowOffset+1, idx2+1, -1.0)
	}
}
