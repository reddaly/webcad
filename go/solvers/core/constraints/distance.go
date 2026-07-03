package constraints

import (
	"fmt"

	"github.com/gonzojive/webcad/schema"
	"gonum.org/v1/gonum/mat"
)

type distanceSubCase int

const (
	distancePtPt distanceSubCase = iota
	distancePtLn
)

type DistanceEvaluator struct {
	subCase  distanceSubCase
	idA, idB string  // idA is always the Point for Pt-Ln
	value    float64 // D
	invC     float64 // 1/C, where C is initial line length squared (for Pt-Ln)
}

func NewDistanceEvaluator(c *schema.Constraint, entities map[string]*schema.Entity) (*DistanceEvaluator, error) {
	d := c.GetDistance()
	idA := d.GetEntityA()
	idB := d.GetEntityB()
	entA, okA := entities[idA]
	entB, okB := entities[idB]
	if !okA || !okB {
		return nil, fmt.Errorf("entities not found: %s, %s", idA, idB)
	}

	isPtA := isPointOrCenter(entA)
	isPtB := isPointOrCenter(entB)
	isLnA := isLine(entA)
	isLnB := isLine(entB)

	if isPtA && isPtB {
		return &DistanceEvaluator{
			subCase: distancePtPt,
			idA:     idA,
			idB:     idB,
			value:   d.GetValue(),
		}, nil
	} else if isPtA && isLnB {
		lineEnt := entB.GetLine()
		dx := lineEnt.X2 - lineEnt.X1
		dy := lineEnt.Y2 - lineEnt.Y1
		C := dx*dx + dy*dy
		if C < 1e-9 {
			C = 1.0
		}
		return &DistanceEvaluator{
			subCase: distancePtLn,
			idA:     idA,
			idB:     idB,
			value:   d.GetValue(),
			invC:    1.0 / C,
		}, nil
	} else if isPtB && isLnA {
		lineEnt := entA.GetLine()
		dx := lineEnt.X2 - lineEnt.X1
		dy := lineEnt.Y2 - lineEnt.Y1
		C := dx*dx + dy*dy
		if C < 1e-9 {
			C = 1.0
		}
		return &DistanceEvaluator{
			subCase: distancePtLn,
			idA:     idB, // Store point in idA
			idB:     idA, // Store line in idB
			value:   d.GetValue(),
			invC:    1.0 / C,
		}, nil
	}

	return nil, fmt.Errorf("unsupported distance configuration between %T and %T", entA.GetEntityType(), entB.GetEntityType())
}

func (d *DistanceEvaluator) NumEquations() int {
	return 1
}

func (d *DistanceEvaluator) EvaluateJacobian(
	x []float64,
	residuals []float64,
	J *mat.Dense,
	rowOffset int,
	paramIndices map[string]int,
) {
	idx1, ok1 := paramIndices[d.idA]
	idx2, ok2 := paramIndices[d.idB]
	if !ok1 || !ok2 {
		return
	}

	switch d.subCase {
	case distancePtPt:
		dx := x[idx1] - x[idx2]
		dy := x[idx1+1] - x[idx2+1]
		dSq := dx*dx + dy*dy

		// r = d^2 - D^2
		residuals[0] = dSq - d.value*d.value

		if J != nil {
			J.Set(rowOffset, idx1, 2.0*dx)
			J.Set(rowOffset, idx1+1, 2.0*dy)
			J.Set(rowOffset, idx2, -2.0*dx)
			J.Set(rowOffset, idx2+1, -2.0*dy)
		}

	case distancePtLn:
		px, py := x[idx1], x[idx1+1]
		x1, y1, x2, y2 := x[idx2], x[idx2+1], x[idx2+2], x[idx2+3]
		dxL := x2 - x1
		dyL := y2 - y1

		num := dyL*(px-x1) - dxL*(py-y1)

		// r = num^2 / C - D^2
		residuals[0] = num*num*d.invC - d.value*d.value

		if J != nil {
			factor := 2.0 * num * d.invC
			J.Set(rowOffset, idx1, factor*dyL)
			J.Set(rowOffset, idx1+1, -factor*dxL)
			J.Set(rowOffset, idx2, factor*(py-y2))
			J.Set(rowOffset, idx2+1, factor*(x2-px))
			J.Set(rowOffset, idx2+2, factor*(y1-py))
			J.Set(rowOffset, idx2+3, factor*(px-x1))
		}
	}
}

func (d *DistanceEvaluator) Evaluate(x []float64, grad []float64, paramIndices map[string]int) float64 {
	idx1, ok1 := paramIndices[d.idA]
	idx2, ok2 := paramIndices[d.idB]
	if !ok1 || !ok2 {
		return 0.0
	}

	switch d.subCase {
	case distancePtPt:
		dx := x[idx1] - x[idx2]
		dy := x[idx1+1] - x[idx2+1]
		dSq := dx*dx + dy*dy

		// r = d^2 - D^2
		r := dSq - d.value*d.value
		valSq := r * r

		if grad != nil {
			factor := 4.0 * r
			grad[idx1] += factor * dx
			grad[idx1+1] += factor * dy
			grad[idx2] -= factor * dx
			grad[idx2+1] -= factor * dy
		}
		return valSq

	case distancePtLn:
		px, py := x[idx1], x[idx1+1]
		x1, y1, x2, y2 := x[idx2], x[idx2+1], x[idx2+2], x[idx2+3]
		dxL := x2 - x1
		dyL := y2 - y1

		num := dyL*(px-x1) - dxL*(py-y1)

		// r = num^2 / C - D^2
		r := num*num*d.invC - d.value*d.value
		valSq := r * r

		if grad != nil {
			factor := 4.0 * r * num * d.invC

			grad[idx1] += factor * dyL
			grad[idx1+1] -= factor * dxL
			grad[idx2] += factor * (py - y2)
			grad[idx2+1] += factor * (x2 - px)
			grad[idx2+2] += factor * (y1 - py)
			grad[idx2+3] += factor * (px - x1)
		}
		return valSq
	}

	return 0.0
}
