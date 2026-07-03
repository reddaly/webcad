package constraints

import (
	"fmt"
	"math"

	"github.com/gonzojive/webcad/schema"
	"gonum.org/v1/gonum/mat"
)

type coincidenceSubCase int

const (
	coincidencePtPt coincidenceSubCase = iota
	coincidencePtLn
	coincidencePtCir
)

type CoincidenceEvaluator struct {
	subCase  coincidenceSubCase
	idA, idB string // idA is always the Point for Pt-Ln and Pt-Cir
	invC     float64
	sqrtInvC float64 // Precomputed sqrt(1/C) for Pt-Ln
}

func NewCoincidenceEvaluator(c *schema.Constraint, entities map[string]*schema.Entity) (*CoincidenceEvaluator, error) {
	cc := c.GetCoincidence()
	idA := cc.GetEntityA()
	idB := cc.GetEntityB()
	entA, okA := entities[idA]
	entB, okB := entities[idB]
	if !okA || !okB {
		return nil, fmt.Errorf("entities not found: %s, %s", idA, idB)
	}

	isPtA := isPoint(entA)
	isPtB := isPoint(entB)
	isLnA := isLine(entA)
	isLnB := isLine(entB)
	isCirA := isCircleOrArc(entA)
	isCirB := isCircleOrArc(entB)

	if isPtA && isPtB {
		return &CoincidenceEvaluator{
			subCase: coincidencePtPt,
			idA:     idA,
			idB:     idB,
		}, nil
	} else if isPtA && isLnB {
		lineEnt := entB.GetLine()
		dx := lineEnt.X2 - lineEnt.X1
		dy := lineEnt.Y2 - lineEnt.Y1
		C := dx*dx + dy*dy
		if C < 1e-9 {
			C = 1.0
		}
		return &CoincidenceEvaluator{
			subCase:  coincidencePtLn,
			idA:      idA,
			idB:      idB,
			invC:     1.0 / C,
			sqrtInvC: math.Sqrt(1.0 / C),
		}, nil
	} else if isPtB && isLnA {
		lineEnt := entA.GetLine()
		dx := lineEnt.X2 - lineEnt.X1
		dy := lineEnt.Y2 - lineEnt.Y1
		C := dx*dx + dy*dy
		if C < 1e-9 {
			C = 1.0
		}
		return &CoincidenceEvaluator{
			subCase:  coincidencePtLn,
			idA:      idB, // Store point in idA
			idB:      idA, // Store line in idB
			invC:     1.0 / C,
			sqrtInvC: math.Sqrt(1.0 / C),
		}, nil
	} else if isPtA && isCirB {
		return &CoincidenceEvaluator{
			subCase: coincidencePtCir,
			idA:     idA,
			idB:     idB,
		}, nil
	} else if isPtB && isCirA {
		return &CoincidenceEvaluator{
			subCase: coincidencePtCir,
			idA:     idB, // Store point in idA
			idB:     idA, // Store circle in idB
		}, nil
	}

	return nil, fmt.Errorf("unsupported coincidence configuration between %T and %T", entA.GetEntityType(), entB.GetEntityType())
}

func (c *CoincidenceEvaluator) NumEquations() int {
	if c.subCase == coincidencePtPt {
		return 2
	}
	return 1
}

func (c *CoincidenceEvaluator) EvaluateJacobian(
	x []float64,
	residuals []float64,
	J *mat.Dense,
	rowOffset int,
	paramIndices map[string]int,
) {
	idx1, ok1 := paramIndices[c.idA]
	idx2, ok2 := paramIndices[c.idB]
	if !ok1 || !ok2 {
		return
	}

	switch c.subCase {
	case coincidencePtPt:
		dx := x[idx1] - x[idx2]
		dy := x[idx1+1] - x[idx2+1]

		residuals[0] = dx
		residuals[1] = dy

		if J != nil {
			// Row 0: r_x = x1 - x2
			J.Set(rowOffset, idx1, 1.0)
			J.Set(rowOffset, idx2, -1.0)

			// Row 1: r_y = y1 - y2
			J.Set(rowOffset+1, idx1+1, 1.0)
			J.Set(rowOffset+1, idx2+1, -1.0)
		}

	case coincidencePtLn:
		px, py := x[idx1], x[idx1+1]
		x1, y1, x2, y2 := x[idx2], x[idx2+1], x[idx2+2], x[idx2+3]
		dxL := x2 - x1
		dyL := y2 - y1

		num := dyL*(px-x1) - dxL*(py-y1)
		residuals[0] = num * c.sqrtInvC

		if J != nil {
			factor := c.sqrtInvC
			J.Set(rowOffset, idx1, factor*dyL)
			J.Set(rowOffset, idx1+1, -factor*dxL)
			J.Set(rowOffset, idx2, factor*(py-y2))
			J.Set(rowOffset, idx2+1, factor*(x2-px))
			J.Set(rowOffset, idx2+2, factor*(y1-py))
			J.Set(rowOffset, idx2+3, factor*(px-x1))
		}

	case coincidencePtCir:
		px, py := x[idx1], x[idx1+1]
		cx, cy, R := x[idx2], x[idx2+1], x[idx2+2]
		dx := cx - px
		dy := cy - py
		dSq := dx*dx + dy*dy

		// r = d^2 - R^2
		residuals[0] = dSq - R*R

		if J != nil {
			J.Set(rowOffset, idx1, -2.0*dx)
			J.Set(rowOffset, idx1+1, -2.0*dy)
			J.Set(rowOffset, idx2, 2.0*dx)
			J.Set(rowOffset, idx2+1, 2.0*dy)
			J.Set(rowOffset, idx2+2, -2.0*R)
		}
	}
}

func (c *CoincidenceEvaluator) Evaluate(x []float64, grad []float64, paramIndices map[string]int) float64 {
	idx1, ok1 := paramIndices[c.idA]
	idx2, ok2 := paramIndices[c.idB]
	if !ok1 || !ok2 {
		return 0.0
	}

	switch c.subCase {
	case coincidencePtPt:
		dx := x[idx1] - x[idx2]
		dy := x[idx1+1] - x[idx2+1]
		if grad != nil {
			grad[idx1] += 2.0 * dx
			grad[idx1+1] += 2.0 * dy
			grad[idx2] -= 2.0 * dx
			grad[idx2+1] -= 2.0 * dy
		}
		return dx*dx + dy*dy

	case coincidencePtLn:
		px, py := x[idx1], x[idx1+1]
		x1, y1, x2, y2 := x[idx2], x[idx2+1], x[idx2+2], x[idx2+3]
		dxL := x2 - x1
		dyL := y2 - y1

		num := dyL*(px-x1) - dxL*(py-y1)

		// E_i = num^2 / C
		valSq := num * num * c.invC

		if grad != nil {
			factor := 2.0 * num * c.invC

			grad[idx1] += factor * dyL
			grad[idx1+1] -= factor * dxL
			grad[idx2] += factor * (py - y2)
			grad[idx2+1] += factor * (x2 - px)
			grad[idx2+2] += factor * (y1 - py)
			grad[idx2+3] += factor * (px - x1)
		}
		return valSq

	case coincidencePtCir:
		px, py := x[idx1], x[idx1+1]
		cx, cy, R := x[idx2], x[idx2+1], x[idx2+2]
		dx := cx - px
		dy := cy - py
		dSq := dx*dx + dy*dy

		// r = d^2 - R^2
		r := dSq - R*R
		valSq := r * r

		if grad != nil {
			factor := 4.0 * r
			grad[idx1] -= factor * dx
			grad[idx1+1] -= factor * dy
			grad[idx2] += factor * dx
			grad[idx2+1] += factor * dy
			grad[idx2+2] -= factor * R
		}
		return valSq
	}

	return 0.0
}
