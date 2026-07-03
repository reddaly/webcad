package constraints

import (
	"fmt"
	"math"

	"github.com/gonzojive/webcad/schema"
	"gonum.org/v1/gonum/mat"
)

type SymmetricEvaluator struct {
	idA, idB, idSym string
}

func NewSymmetricEvaluator(c *schema.Constraint, entities map[string]*schema.Entity) (*SymmetricEvaluator, error) {
	s := c.GetSymmetric()
	idA := s.GetEntityA()
	idB := s.GetEntityB()
	idSym := s.GetSymmetryLine()
	if _, ok := entities[idA]; !ok {
		return nil, fmt.Errorf("entity A %s not found", idA)
	}
	if _, ok := entities[idB]; !ok {
		return nil, fmt.Errorf("entity B %s not found", idB)
	}
	if _, ok := entities[idSym]; !ok {
		return nil, fmt.Errorf("symmetry line %s not found", idSym)
	}
	return &SymmetricEvaluator{
		idA:   idA,
		idB:   idB,
		idSym: idSym,
	}, nil
}

func (s *SymmetricEvaluator) NumEquations() int {
	return 2
}

func (s *SymmetricEvaluator) EvaluateJacobian(
	x []float64,
	residuals []float64,
	J *mat.Dense,
	rowOffset int,
	paramIndices map[string]int,
) {
	idxA, ok1 := paramIndices[s.idA]
	idxB, ok2 := paramIndices[s.idB]
	idxSym, ok3 := paramIndices[s.idSym]
	if !ok1 || !ok2 || !ok3 {
		return
	}

	xa, ya := x[idxA], x[idxA+1]
	xb, yb := x[idxB], x[idxB+1]
	x1_val, y1_val, x2_val, y2_val := x[idxSym], x[idxSym+1], x[idxSym+2], x[idxSym+3]

	dx := x2_val - x1_val
	dy := y2_val - y1_val
	den := dx*dx + dy*dy

	if den > 1e-9 {
		numMid := dy*(xa+xb-2.0*x1_val) - dx*(ya+yb-2.0*y1_val)
		rMid := numMid / (2.0 * math.Sqrt(den))

		dot := (xb-xa)*dx + (yb-ya)*dy
		rPerp := dot / math.Sqrt(den)

		residuals[0] = rMid
		residuals[1] = rPerp

		if J != nil {
			invSqrtD := 1.0 / math.Sqrt(den)
			factorMid := numMid / (2.0 * den)

			dNm_dx1 := -2.0*dy + (ya + yb - 2.0*y1_val)
			dNm_dy1 := 2.0*dx - (xa + xb - 2.0*x1_val)
			dNm_dx2 := -(ya + yb - 2.0*y1_val)
			dNm_dy2 := xa + xb - 2.0*x1_val

			drMid_dxa := dy * 0.5 * invSqrtD
			drMid_dxb := dy * 0.5 * invSqrtD
			drMid_dya := -dx * 0.5 * invSqrtD
			drMid_dyb := -dx * 0.5 * invSqrtD

			drMid_dx1 := (dNm_dx1 + factorMid*2.0*dx) * 0.5 * invSqrtD
			drMid_dy1 := (dNm_dy1 + factorMid*2.0*dy) * 0.5 * invSqrtD
			drMid_dx2 := (dNm_dx2 - factorMid*2.0*dx) * 0.5 * invSqrtD
			drMid_dy2 := (dNm_dy2 - factorMid*2.0*dy) * 0.5 * invSqrtD

			factorPerp := dot / (2.0 * den)

			drPerp_dxa := -dx * invSqrtD
			drPerp_dxb := dx * invSqrtD
			drPerp_dya := -dy * invSqrtD
			drPerp_dyb := dy * invSqrtD

			drPerp_dx1 := (-(xb - xa) + factorPerp*2.0*dx) * invSqrtD
			drPerp_dy1 := (-(yb - ya) + factorPerp*2.0*dy) * invSqrtD
			drPerp_dx2 := (xb - xa - factorPerp*2.0*dx) * invSqrtD
			drPerp_dy2 := (yb - ya - factorPerp*2.0*dy) * invSqrtD

			// Row 0: rMid
			J.Set(rowOffset, idxA, drMid_dxa)
			J.Set(rowOffset, idxA+1, drMid_dya)
			J.Set(rowOffset, idxB, drMid_dxb)
			J.Set(rowOffset, idxB+1, drMid_dyb)
			J.Set(rowOffset, idxSym, drMid_dx1)
			J.Set(rowOffset, idxSym+1, drMid_dy1)
			J.Set(rowOffset, idxSym+2, drMid_dx2)
			J.Set(rowOffset, idxSym+3, drMid_dy2)

			// Row 1: rPerp
			J.Set(rowOffset+1, idxA, drPerp_dxa)
			J.Set(rowOffset+1, idxA+1, drPerp_dya)
			J.Set(rowOffset+1, idxB, drPerp_dxb)
			J.Set(rowOffset+1, idxB+1, drPerp_dyb)
			J.Set(rowOffset+1, idxSym, drPerp_dx1)
			J.Set(rowOffset+1, idxSym+1, drPerp_dy1)
			J.Set(rowOffset+1, idxSym+2, drPerp_dx2)
			J.Set(rowOffset+1, idxSym+3, drPerp_dy2)
		}
	} else {
		// Collapsed line fallback (Point-Point coincidence between A and B)
		dxAB := xb - xa
		dyAB := yb - ya

		residuals[0] = dxAB
		residuals[1] = dyAB

		if J != nil {
			// Row 0: r_x = xa - xb
			J.Set(rowOffset, idxA, 1.0)
			J.Set(rowOffset, idxB, -1.0)

			// Row 1: r_y = ya - yb
			J.Set(rowOffset+1, idxA+1, 1.0)
			J.Set(rowOffset+1, idxB+1, -1.0)
		}
	}
}

func (s *SymmetricEvaluator) Evaluate(x []float64, grad []float64, paramIndices map[string]int) float64 {
	idxA, ok1 := paramIndices[s.idA]
	idxB, ok2 := paramIndices[s.idB]
	idxSym, ok3 := paramIndices[s.idSym]
	if !ok1 || !ok2 || !ok3 {
		return 0.0
	}

	xa, ya := x[idxA], x[idxA+1]
	xb, yb := x[idxB], x[idxB+1]
	x1_val, y1_val, x2_val, y2_val := x[idxSym], x[idxSym+1], x[idxSym+2], x[idxSym+3]

	dx := x2_val - x1_val
	dy := y2_val - y1_val
	den := dx*dx + dy*dy

	if den > 1e-9 {
		numMid := dy*(xa+xb-2.0*x1_val) - dx*(ya+yb-2.0*y1_val)
		rMid := numMid / (2.0 * math.Sqrt(den))

		dot := (xb-xa)*dx + (yb-ya)*dy
		rPerp := dot / math.Sqrt(den)

		totalResidualSq := rMid*rMid + rPerp*rPerp

		if grad != nil {
			invSqrtD := 1.0 / math.Sqrt(den)
			factorMid := numMid / (2.0 * den)

			dNm_dx1 := -2.0*dy + (ya + yb - 2.0*y1_val)
			dNm_dy1 := 2.0*dx - (xa + xb - 2.0*x1_val)
			dNm_dx2 := -(ya + yb - 2.0*y1_val)
			dNm_dy2 := xa + xb - 2.0*x1_val

			drMid_dxa := dy * 0.5 * invSqrtD
			drMid_dxb := dy * 0.5 * invSqrtD
			drMid_dya := -dx * 0.5 * invSqrtD
			drMid_dyb := -dx * 0.5 * invSqrtD

			drMid_dx1 := (dNm_dx1 + factorMid*2.0*dx) * 0.5 * invSqrtD
			drMid_dy1 := (dNm_dy1 + factorMid*2.0*dy) * 0.5 * invSqrtD
			drMid_dx2 := (dNm_dx2 - factorMid*2.0*dx) * 0.5 * invSqrtD
			drMid_dy2 := (dNm_dy2 - factorMid*2.0*dy) * 0.5 * invSqrtD

			factorPerp := dot / (2.0 * den)

			drPerp_dxa := -dx * invSqrtD
			drPerp_dxb := dx * invSqrtD
			drPerp_dya := -dy * invSqrtD
			drPerp_dyb := dy * invSqrtD

			drPerp_dx1 := (-(xb - xa) + factorPerp*2.0*dx) * invSqrtD
			drPerp_dy1 := (-(yb - ya) + factorPerp*2.0*dy) * invSqrtD
			drPerp_dx2 := (xb - xa - factorPerp*2.0*dx) * invSqrtD
			drPerp_dy2 := (yb - ya - factorPerp*2.0*dy) * invSqrtD

			r2Mid := 2.0 * rMid
			r2Perp := 2.0 * rPerp

			grad[idxA] += r2Mid*drMid_dxa + r2Perp*drPerp_dxa
			grad[idxA+1] += r2Mid*drMid_dya + r2Perp*drPerp_dya

			grad[idxB] += r2Mid*drMid_dxb + r2Perp*drPerp_dxb
			grad[idxB+1] += r2Mid*drMid_dyb + r2Perp*drPerp_dyb

			grad[idxSym] += r2Mid*drMid_dx1 + r2Perp*drPerp_dx1
			grad[idxSym+1] += r2Mid*drMid_dy1 + r2Perp*drPerp_dy1
			grad[idxSym+2] += r2Mid*drMid_dx2 + r2Perp*drPerp_dx2
			grad[idxSym+3] += r2Mid*drMid_dy2 + r2Perp*drPerp_dy2
		}
		return totalResidualSq
	} else {
		dxAB := xb - xa
		dyAB := yb - ya
		totalResidualSq := dxAB*dxAB + dyAB*dyAB
		if grad != nil {
			grad[idxA] += 2.0 * dxAB
			grad[idxA+1] += 2.0 * dyAB
			grad[idxB] -= 2.0 * dxAB
			grad[idxB+1] -= 2.0 * dyAB
		}
		return totalResidualSq
	}
}
