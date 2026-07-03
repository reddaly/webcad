package constraints

import (
	"fmt"
	"math"

	"github.com/gonzojive/webcad/schema"
	"gonum.org/v1/gonum/mat"
)

type AngleEvaluator struct {
	idA, idB  string
	sinTarget float64
	cosTarget float64
}

func NewAngleEvaluator(c *schema.Constraint, entities map[string]*schema.Entity) (*AngleEvaluator, error) {
	a := c.GetAngle()
	idA := a.GetEntityA()
	idB := a.GetEntityB()
	if _, ok := entities[idA]; !ok {
		return nil, fmt.Errorf("entity A %s not found", idA)
	}
	if _, ok := entities[idB]; !ok {
		return nil, fmt.Errorf("entity B %s not found", idB)
	}

	target := a.GetValueRadians()
	return &AngleEvaluator{
		idA:       idA,
		idB:       idB,
		sinTarget: math.Sin(target),
		cosTarget: math.Cos(target),
	}, nil
}

func (a *AngleEvaluator) NumEquations() int {
	return 1
}

func (a *AngleEvaluator) EvaluateJacobian(
	x []float64,
	residuals []float64,
	J *mat.Dense,
	rowOffset int,
	paramIndices map[string]int,
) {
	idx1, ok1 := paramIndices[a.idA]
	idx2, ok2 := paramIndices[a.idB]
	if !ok1 || !ok2 {
		return
	}

	x1_val, y1_val, x2_val, y2_val := x[idx1], x[idx1+1], x[idx1+2], x[idx1+3]
	x3_val, y3_val, x4_val, y4_val := x[idx2], x[idx2+1], x[idx2+2], x[idx2+3]

	v1x, v1y := x2_val-x1_val, y2_val-y1_val
	v2x, v2y := x4_val-x3_val, y4_val-y3_val

	dot := v1x*v2x + v1y*v2y
	cross := v1x*v2y - v1y*v2x

	// r = (v1 . v2)*sin(theta_T) - (v1 x v2)*cos(theta_T)
	residuals[0] = dot*a.sinTarget - cross*a.cosTarget

	if J != nil {
		dr_dv1x := v2x*a.sinTarget - v2y*a.cosTarget
		dr_dv1y := v2y*a.sinTarget + v2x*a.cosTarget
		dr_dv2x := v1x*a.sinTarget + v1y*a.cosTarget
		dr_dv2y := v1y*a.sinTarget - v1x*a.cosTarget

		// v1 = p2 - p1 => dv1/dp2 = 1, dv1/dp1 = -1
		// v2 = p4 - p3 => dv2/dp4 = 1, dv2/dp3 = -1
		J.Set(rowOffset, idx1, -dr_dv1x)
		J.Set(rowOffset, idx1+1, -dr_dv1y)
		J.Set(rowOffset, idx1+2, dr_dv1x)
		J.Set(rowOffset, idx1+3, dr_dv1y)

		J.Set(rowOffset, idx2, -dr_dv2x)
		J.Set(rowOffset, idx2+1, -dr_dv2y)
		J.Set(rowOffset, idx2+2, dr_dv2x)
		J.Set(rowOffset, idx2+3, dr_dv2y)
	}
}

func (a *AngleEvaluator) Evaluate(x []float64, grad []float64, paramIndices map[string]int) float64 {
	idx1, ok1 := paramIndices[a.idA]
	idx2, ok2 := paramIndices[a.idB]
	if !ok1 || !ok2 {
		return 0.0
	}

	x1_val, y1_val, x2_val, y2_val := x[idx1], x[idx1+1], x[idx1+2], x[idx1+3]
	x3_val, y3_val, x4_val, y4_val := x[idx2], x[idx2+1], x[idx2+2], x[idx2+3]

	v1x, v1y := x2_val-x1_val, y2_val-y1_val
	v2x, v2y := x4_val-x3_val, y4_val-y3_val

	dot := v1x*v2x + v1y*v2y
	cross := v1x*v2y - v1y*v2x

	// r = (v1 . v2)*sin(theta_T) - (v1 x v2)*cos(theta_T)
	r := dot*a.sinTarget - cross*a.cosTarget
	totalResidualSq := r * r

	if grad != nil {
		factor := 2.0 * r

		// Derivatives of r w.r.t v1x, v1y, v2x, v2y
		dr_dv1x := v2x*a.sinTarget - v2y*a.cosTarget
		dr_dv1y := v2y*a.sinTarget + v2x*a.cosTarget
		dr_dv2x := v1x*a.sinTarget + v1y*a.cosTarget
		dr_dv2y := v1y*a.sinTarget - v1x*a.cosTarget

		// v1 = p2 - p1 => dv1/dp2 = 1, dv1/dp1 = -1
		// v2 = p4 - p3 => dv2/dp4 = 1, dv2/dp3 = -1

		grad[idx1] -= factor * dr_dv1x
		grad[idx1+1] -= factor * dr_dv1y
		grad[idx1+2] += factor * dr_dv1x
		grad[idx1+3] += factor * dr_dv1y

		grad[idx2] -= factor * dr_dv2x
		grad[idx2+1] -= factor * dr_dv2y
		grad[idx2+2] += factor * dr_dv2x
		grad[idx2+3] += factor * dr_dv2y
	}

	return totalResidualSq
}
