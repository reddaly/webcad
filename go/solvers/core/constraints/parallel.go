package constraints

import (
	"fmt"
	"math"

	"github.com/gonzojive/webcad/schema"
	"gonum.org/v1/gonum/mat"
)

type ParallelEvaluator struct {
	idA, idB string
	normC    float64
}

func NewParallelEvaluator(c *schema.Constraint, entities map[string]*schema.Entity) (*ParallelEvaluator, error) {
	p := c.GetParallel()
	idA := p.GetLineA()
	idB := p.GetLineB()
	entA, okA := entities[idA]
	entB, okB := entities[idB]
	if !okA || !okB {
		return nil, fmt.Errorf("entities not found: %s, %s", idA, idB)
	}

	// Compute constant normalization using initial lengths
	paramsA := getParams(entA)
	paramsB := getParams(entB)
	if len(paramsA) < 4 || len(paramsB) < 4 {
		return nil, fmt.Errorf("invalid line parameters")
	}

	ilen1Sq := (paramsA[2]-paramsA[0])*(paramsA[2]-paramsA[0]) + (paramsA[3]-paramsA[1])*(paramsA[3]-paramsA[1])
	ilen2Sq := (paramsB[2]-paramsB[0])*(paramsB[2]-paramsB[0]) + (paramsB[3]-paramsB[1])*(paramsB[3]-paramsB[1])
	if ilen1Sq < 1e-9 {
		ilen1Sq = 1.0
	}
	if ilen2Sq < 1e-9 {
		ilen2Sq = 1.0
	}
	normC := math.Sqrt(ilen1Sq * ilen2Sq)

	return &ParallelEvaluator{
		idA:   idA,
		idB:   idB,
		normC: normC,
	}, nil
}

func (p *ParallelEvaluator) NumEquations() int {
	return 1
}

func (p *ParallelEvaluator) EvaluateJacobian(
	x []float64,
	residuals []float64,
	J *mat.Dense,
	rowOffset int,
	paramIndices map[string]int,
) {
	idx1, ok1 := paramIndices[p.idA]
	idx2, ok2 := paramIndices[p.idB]
	if !ok1 || !ok2 {
		return
	}

	x1, y1, x2, y2 := x[idx1], x[idx1+1], x[idx1+2], x[idx1+3]
	x3, y3, x4, y4 := x[idx2], x[idx2+1], x[idx2+2], x[idx2+3]

	dx1, dy1 := x2-x1, y2-y1
	dx2, dy2 := x4-x3, y4-y3
	cross := dx1*dy2 - dy1*dx2

	// r = cross / normC
	residuals[0] = cross / p.normC

	if J != nil {
		invC := 1.0 / p.normC
		// Row 0 of this constraint's Jacobian block
		J.Set(rowOffset, idx1, -dy2*invC)
		J.Set(rowOffset, idx1+1, dx2*invC)
		J.Set(rowOffset, idx1+2, dy2*invC)
		J.Set(rowOffset, idx1+3, -dx2*invC)

		J.Set(rowOffset, idx2, dy1*invC)
		J.Set(rowOffset, idx2+1, -dx1*invC)
		J.Set(rowOffset, idx2+2, -dy1*invC)
		J.Set(rowOffset, idx2+3, dx1*invC)
	}
}

func (p *ParallelEvaluator) Evaluate(x []float64, grad []float64, paramIndices map[string]int) float64 {
	idx1, ok1 := paramIndices[p.idA]
	idx2, ok2 := paramIndices[p.idB]
	if !ok1 || !ok2 {
		return 0.0
	}

	x1, y1, x2, y2 := x[idx1], x[idx1+1], x[idx1+2], x[idx1+3]
	x3, y3, x4, y4 := x[idx2], x[idx2+1], x[idx2+2], x[idx2+3]

	dx1, dy1 := x2-x1, y2-y1
	dx2, dy2 := x4-x3, y4-y3
	cross := dx1*dy2 - dy1*dx2
	r := cross / p.normC
	totalResidualSq := r * r

	if grad != nil {
		factor := 2.0 * r / p.normC
		grad[idx1] -= factor * dy2
		grad[idx1+1] += factor * dx2
		grad[idx1+2] += factor * dy2
		grad[idx1+3] -= factor * dx2

		grad[idx2] += factor * dy1
		grad[idx2+1] -= factor * dx1
		grad[idx2+2] -= factor * dy1
		grad[idx2+3] += factor * dx1
	}
	return totalResidualSq
}
