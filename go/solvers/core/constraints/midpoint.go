package constraints

import (
	"fmt"
	"github.com/gonzojive/webcad/schema"
	"gonum.org/v1/gonum/mat"
)

type MidpointEvaluator struct {
	idPt, idLn string
}

func NewMidpointEvaluator(c *schema.Constraint, entities map[string]*schema.Entity) (*MidpointEvaluator, error) {
	m := c.GetMidpoint()
	idPt := m.GetPoint()
	idLn := m.GetLine()
	entPt, okPt := entities[idPt]
	if !okPt {
		return nil, fmt.Errorf("point entity %s not found", idPt)
	}
	if !isPoint(entPt) {
		return nil, fmt.Errorf("entity %s is not a point", idPt)
	}
	entLn, okLn := entities[idLn]
	if !okLn {
		return nil, fmt.Errorf("line entity %s not found", idLn)
	}
	if !isLine(entLn) {
		return nil, fmt.Errorf("entity %s is not a line", idLn)
	}
	return &MidpointEvaluator{idPt: idPt, idLn: idLn}, nil
}

func (m *MidpointEvaluator) Evaluate(x []float64, grad []float64, paramIndices map[string]int) float64 {
	idxPt, ok1 := paramIndices[m.idPt]
	idxLn, ok2 := paramIndices[m.idLn]
	if !ok1 || !ok2 {
		return 0.0
	}

	px, py := x[idxPt], x[idxPt+1]
	x1, y1, x2, y2 := x[idxLn], x[idxLn+1], x[idxLn+2], x[idxLn+3]

	mx := 0.5 * (x1 + x2)
	my := 0.5 * (y1 + y2)

	rx := px - mx
	ry := py - my
	totalResidualSq := rx*rx + ry*ry

	if grad != nil {
		grad[idxPt] += 2.0 * rx
		grad[idxPt+1] += 2.0 * ry

		grad[idxLn] -= rx
		grad[idxLn+1] -= ry
		grad[idxLn+2] -= rx
		grad[idxLn+3] -= ry
	}
	return totalResidualSq
}

func (m *MidpointEvaluator) NumEquations() int {
	return 2
}

func (m *MidpointEvaluator) EvaluateJacobian(x []float64, residuals []float64, J *mat.Dense, rowOffset int, paramIndices map[string]int) {
	idxPt, ok1 := paramIndices[m.idPt]
	idxLn, ok2 := paramIndices[m.idLn]
	if !ok1 || !ok2 {
		return
	}

	px, py := x[idxPt], x[idxPt+1]
	x1, y1, x2, y2 := x[idxLn], x[idxLn+1], x[idxLn+2], x[idxLn+3]

	residuals[0] = px - 0.5*(x1+x2)
	residuals[1] = py - 0.5*(y1+y2)

	if J != nil {
		// Row 0: r_0 = px - 0.5*x1 - 0.5*x2
		J.Set(rowOffset, idxPt, 1.0)
		J.Set(rowOffset, idxLn, -0.5)
		J.Set(rowOffset, idxLn+2, -0.5)

		// Row 1: r_1 = py - 0.5*y1 - 0.5*y2
		J.Set(rowOffset+1, idxPt+1, 1.0)
		J.Set(rowOffset+1, idxLn+1, -0.5)
		J.Set(rowOffset+1, idxLn+3, -0.5)
	}
}
