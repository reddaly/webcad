package constraints

import (
	"fmt"
	"math"

	"github.com/gonzojive/webcad/schema"
	"gonum.org/v1/gonum/mat"
)

type tangentSubCase int

const (
	tangentCirCir tangentSubCase = iota
	tangentCirLn
)

type TangentEvaluator struct {
	subCase    tangentSubCase
	idA, idB   string // idA is always the Circle for Cir-Ln
	isInternal bool   // For Cir-Cir
	invC       float64
}

func NewTangentEvaluator(c *schema.Constraint, entities map[string]*schema.Entity) (*TangentEvaluator, error) {
	t := c.GetTangent()
	idA := t.GetEntityA()
	idB := t.GetEntityB()
	entA, okA := entities[idA]
	entB, okB := entities[idB]
	if !okA || !okB {
		return nil, fmt.Errorf("entities not found: %s, %s", idA, idB)
	}

	isCirA := isCircleOrArc(entA)
	isCirB := isCircleOrArc(entB)
	isLnA := isLine(entA)
	isLnB := isLine(entB)

	if isCirA && isCirB {
		// Determine and lock chirality based on initial state
		paramsA := getParams(entA)
		paramsB := getParams(entB)
		if len(paramsA) < 3 || len(paramsB) < 3 {
			return nil, fmt.Errorf("invalid circle parameters")
		}
		cxA, cyA, rA := paramsA[0], paramsA[1], paramsA[2]
		cxB, cyB, rB := paramsB[0], paramsB[1], paramsB[2]

		dx := cxA - cxB
		dy := cyA - cyB
		d := math.Sqrt(dx*dx + dy*dy)

		extErr := math.Abs(d - (rA + rB))
		intErr := math.Abs(d - math.Abs(rA-rB))

		isInternal := intErr < extErr

		return &TangentEvaluator{
			subCase:    tangentCirCir,
			idA:        idA,
			idB:        idB,
			isInternal: isInternal,
		}, nil
	} else if isCirA && isLnB {
		lineEnt := entB.GetLine()
		dx := lineEnt.X2 - lineEnt.X1
		dy := lineEnt.Y2 - lineEnt.Y1
		C := dx*dx + dy*dy
		if C < 1e-9 {
			C = 1.0
		}
		return &TangentEvaluator{
			subCase: tangentCirLn,
			idA:     idA,
			idB:     idB,
			invC:    1.0 / C,
		}, nil
	} else if isCirB && isLnA {
		lineEnt := entA.GetLine()
		dx := lineEnt.X2 - lineEnt.X1
		dy := lineEnt.Y2 - lineEnt.Y1
		C := dx*dx + dy*dy
		if C < 1e-9 {
			C = 1.0
		}
		return &TangentEvaluator{
			subCase: tangentCirLn,
			idA:     idB, // Store circle in idA
			idB:     idA, // Store line in idB
			invC:    1.0 / C,
		}, nil
	}

	return nil, fmt.Errorf("unsupported tangent configuration between %T and %T", entA.GetEntityType(), entB.GetEntityType())
}

func (t *TangentEvaluator) NumEquations() int {
	return 1
}

func (t *TangentEvaluator) EvaluateJacobian(
	x []float64,
	residuals []float64,
	J *mat.Dense,
	rowOffset int,
	paramIndices map[string]int,
) {
	idx1, ok1 := paramIndices[t.idA]
	idx2, ok2 := paramIndices[t.idB]
	if !ok1 || !ok2 {
		return
	}

	switch t.subCase {
	case tangentCirCir:
		cx1, cy1, r1 := x[idx1], x[idx1+1], x[idx1+2]
		cx2, cy2, r2 := x[idx2], x[idx2+1], x[idx2+2]
		dx := cx1 - cx2
		dy := cy1 - cy2
		dSq := dx*dx + dy*dy

		var S float64
		if t.isInternal {
			S = r1 - r2
		} else {
			S = r1 + r2
		}
		SSq := S * S

		// r = d^2 - (r1 \pm r2)^2
		residuals[0] = dSq - SSq

		if J != nil {
			J.Set(rowOffset, idx1, 2.0*dx)
			J.Set(rowOffset, idx1+1, 2.0*dy)
			J.Set(rowOffset, idx2, -2.0*dx)
			J.Set(rowOffset, idx2+1, -2.0*dy)

			J.Set(rowOffset, idx1+2, -2.0*S)
			if t.isInternal {
				J.Set(rowOffset, idx2+2, 2.0*S)
			} else {
				J.Set(rowOffset, idx2+2, -2.0*S)
			}
		}

	case tangentCirLn:
		cx, cy, R := x[idx1], x[idx1+1], x[idx1+2]
		x1, y1, x2, y2 := x[idx2], x[idx2+1], x[idx2+2], x[idx2+3]
		dxL := x2 - x1
		dyL := y2 - y1
		C := dxL*dxL + dyL*dyL
		if C < 1e-9 {
			C = 1.0
		}
		invC := 1.0 / C

		num := dyL*(cx-x1) - dxL*(cy-y1)

		// r = num^2 / C - R^2
		residuals[0] = num*num*invC - R*R

		if J != nil {
			factor := 2.0 * num * invC
			factorK := factor * num * invC
			J.Set(rowOffset, idx1, factor*dyL)
			J.Set(rowOffset, idx1+1, -factor*dxL)
			J.Set(rowOffset, idx2, factor*(cy-y2)+factorK*dxL)
			J.Set(rowOffset, idx2+1, factor*(x2-cx)+factorK*dyL)
			J.Set(rowOffset, idx2+2, factor*(y1-cy)-factorK*dxL)
			J.Set(rowOffset, idx2+3, factor*(cx-x1)-factorK*dyL)

			J.Set(rowOffset, idx1+2, -2.0*R)
		}
	}
}

func (t *TangentEvaluator) Evaluate(x []float64, grad []float64, paramIndices map[string]int) float64 {
	idx1, ok1 := paramIndices[t.idA]
	idx2, ok2 := paramIndices[t.idB]
	if !ok1 || !ok2 {
		return 0.0
	}

	switch t.subCase {
	case tangentCirCir:
		cx1, cy1, r1 := x[idx1], x[idx1+1], x[idx1+2]
		cx2, cy2, r2 := x[idx2], x[idx2+1], x[idx2+2]
		dx := cx1 - cx2
		dy := cy1 - cy2
		dSq := dx*dx + dy*dy

		var S float64
		if t.isInternal {
			S = r1 - r2
		} else {
			S = r1 + r2
		}
		SSq := S * S

		// r = d^2 - (r1 \pm r2)^2
		r := dSq - SSq
		totalResidualSq := r * r

		if grad != nil {
			factor := 2.0 * r

			// dr/dcx1 = 2*dx, dr/dcy1 = 2*dy
			// dr/dcx2 = -2*dx, dr/dcy2 = -2*dy
			dr_dcx1 := 2.0 * dx
			dr_dcy1 := 2.0 * dy

			grad[idx1] += factor * dr_dcx1
			grad[idx1+1] += factor * dr_dcy1
			grad[idx2] -= factor * dr_dcx1
			grad[idx2+1] -= factor * dr_dcy1

			// dr/dr1 = -2*S
			// dr/dr2 = -2*S (external) or 2*S (internal)
			dr_dr1 := -2.0 * S
			var dr_dr2 float64
			if t.isInternal {
				dr_dr2 = 2.0 * S
			} else {
				dr_dr2 = -2.0 * S
			}

			grad[idx1+2] += factor * dr_dr1
			grad[idx2+2] += factor * dr_dr2
		}
		return totalResidualSq

	case tangentCirLn:
		cx, cy, R := x[idx1], x[idx1+1], x[idx1+2]
		x1, y1, x2, y2 := x[idx2], x[idx2+1], x[idx2+2], x[idx2+3]
		dxL := x2 - x1
		dyL := y2 - y1
		C := dxL*dxL + dyL*dyL
		if C < 1e-9 {
			C = 1.0
		}
		invC := 1.0 / C

		num := dyL*(cx-x1) - dxL*(cy-y1)

		// r = num^2 / C - R^2
		r := num*num*invC - R*R
		valSq := r * r

		if grad != nil {
			factor := 4.0 * r * num * invC
			factorK := 2.0 * r * (num * invC) * (num * invC)

			grad[idx1] += factor * dyL
			grad[idx1+1] -= factor * dxL
			grad[idx2] += factor*(cy-y2) + 2.0*factorK*dxL
			grad[idx2+1] += factor*(x2-cx) + 2.0*factorK*dyL
			grad[idx2+2] += factor*(y1-cy) - 2.0*factorK*dxL
			grad[idx2+3] += factor*(cx-x1) - 2.0*factorK*dyL

			grad[idx1+2] -= 4.0 * r * R
		}
		return valSq
	}

	return 0.0
}
