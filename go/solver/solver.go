// Package solver provides a geometric constraint solving engine.
package solver

import (
	"math"

	"gonum.org/v1/gonum/optimize"
)

// SolverAlgorithm defines the algorithm used for optimization.
type SolverAlgorithm string

const (
	// AlgorithmBFGS represents the Broyden-Fletcher-Goldfarb-Shanno algorithm.
	AlgorithmBFGS SolverAlgorithm = "bfgs"
	// AlgorithmLM represents the Levenberg-Marquardt algorithm.
	AlgorithmLM SolverAlgorithm = "lm"
)

// problemState holds the optimization variables and mappings.
type problemState struct {
	initialX []float64
	pointIdx map[EntityID]int
	circleIdx map[EntityID]int
}

// Solve resolves the sketch constraints to find the optimal point positions.
func Solve(state SketchState, algo SolverAlgorithm) *SolverResult {
	prob := buildProblemState(&state)

	objFunc := func(x []float64) float64 {
		return evaluateConstraints(&state, prob, x)
	}

	resultX, err := runOptimization(objFunc, prob.initialX, algo)
	if err != nil {
		return &SolverResult{
			Success: false,
			Error:   err.Error(),
		}
	}

	return reconstructState(&state, prob, resultX)
}

func buildProblemState(state *SketchState) *problemState {
	prob := &problemState{
		pointIdx:  make(map[EntityID]int),
		circleIdx: make(map[EntityID]int),
	}

	for _, p := range state.Points {
		if !p.Fixed {
			prob.pointIdx[p.ID] = len(prob.initialX)
			prob.initialX = append(prob.initialX, p.X, p.Y)
		}
	}

	for _, c := range state.Circles {
		if !c.FixedRadius {
			prob.circleIdx[c.ID] = len(prob.initialX)
			prob.initialX = append(prob.initialX, c.Radius)
		}
	}
	return prob
}

func runOptimization(objFunc func([]float64) float64, initialX []float64, algo SolverAlgorithm) ([]float64, error) {
	if len(initialX) == 0 {
		return initialX, nil
	}

	problem := optimize.Problem{
		Func: objFunc,
	}

	var method optimize.Method
	if algo == AlgorithmLM {
		method = &optimize.BFGS{}
	} else {
		method = &optimize.BFGS{}
	}

	settings := &optimize.Settings{
		GradientThreshold: 1e-6,
	}
	result, err := optimize.Minimize(problem, initialX, settings, method)
	if err != nil {
		return nil, err
	}
	return result.X, nil
}

func reconstructState(state *SketchState, prob *problemState, resultX []float64) *SolverResult {
	res := &SolverResult{
		Success: true,
	}
	for _, p := range state.Points {
		if idx, ok := prob.pointIdx[p.ID]; ok {
			p.X = resultX[idx]
			p.Y = resultX[idx+1]
		}
		res.Points = append(res.Points, p)
	}

	for _, c := range state.Circles {
		if idx, ok := prob.circleIdx[c.ID]; ok {
			c.Radius = resultX[idx]
		}
		res.Circles = append(res.Circles, c)
	}
	return res
}

func evaluateConstraints(state *SketchState, prob *problemState, x []float64) float64 {
	var totalError float64
	for _, c := range state.Constraints {
		totalError += evaluateConstraint(state, prob, x, c)
	}
	return totalError
}

func evaluateConstraint(state *SketchState, prob *problemState, x []float64, c Constraint) float64 {
	switch c.Type {
	case ConstraintCoincident:
		return evalCoincident(state, prob, x, c)
	case ConstraintDistance:
		return evalDistance(state, prob, x, c)
	case ConstraintHorizontalDistance:
		return evalHorizontalDistance(state, prob, x, c)
	case ConstraintVerticalDistance:
		return evalVerticalDistance(state, prob, x, c)
	case ConstraintHorizontal:
		return evalHorizontal(state, prob, x, c)
	case ConstraintVertical:
		return evalVertical(state, prob, x, c)
	case ConstraintParallel:
		return evalParallel(state, prob, x, c)
	case ConstraintPerpendicular:
		return evalPerpendicular(state, prob, x, c)
	case ConstraintPointLineDistance:
		return evalPointLineDistance(state, prob, x, c)
	}
	return 0
}

func getPoint(state *SketchState, prob *problemState, id EntityID, x []float64) (float64, float64) {
	if idx, ok := prob.pointIdx[id]; ok {
		return x[idx], x[idx+1]
	}
	for _, p := range state.Points {
		if p.ID == id {
			return p.X, p.Y
		}
	}
	return 0, 0
}

func getLine(state *SketchState, id EntityID) *Line {
	for _, l := range state.Lines {
		if l.ID == id {
			return &l
		}
	}
	return nil
}

func evalCoincident(state *SketchState, prob *problemState, x []float64, c Constraint) float64 {
	if len(c.EntityIDs) != 2 {
		return 0
	}
	x1, y1 := getPoint(state, prob, c.EntityIDs[0], x)
	x2, y2 := getPoint(state, prob, c.EntityIDs[1], x)
	dx, dy := x1-x2, y1-y2
	return dx*dx + dy*dy
}

func evalDistance(state *SketchState, prob *problemState, x []float64, c Constraint) float64 {
	if len(c.EntityIDs) != 2 {
		return 0
	}
	x1, y1 := getPoint(state, prob, c.EntityIDs[0], x)
	x2, y2 := getPoint(state, prob, c.EntityIDs[1], x)
	dx, dy := x1-x2, y1-y2
	dist := math.Sqrt(dx*dx + dy*dy)
	err := dist - c.Value
	return err * err
}

func evalHorizontalDistance(state *SketchState, prob *problemState, x []float64, c Constraint) float64 {
	if len(c.EntityIDs) != 2 {
		return 0
	}
	x1, _ := getPoint(state, prob, c.EntityIDs[0], x)
	x2, _ := getPoint(state, prob, c.EntityIDs[1], x)
	err := math.Abs(x1-x2) - c.Value
	return err * err
}

func evalVerticalDistance(state *SketchState, prob *problemState, x []float64, c Constraint) float64 {
	if len(c.EntityIDs) != 2 {
		return 0
	}
	_, y1 := getPoint(state, prob, c.EntityIDs[0], x)
	_, y2 := getPoint(state, prob, c.EntityIDs[1], x)
	err := math.Abs(y1-y2) - c.Value
	return err * err
}

func evalHorizontal(state *SketchState, prob *problemState, x []float64, c Constraint) float64 {
	if len(c.EntityIDs) != 1 {
		return 0
	}
	l := getLine(state, c.EntityIDs[0])
	if l == nil {
		return 0
	}
	_, y1 := getPoint(state, prob, l.P1ID, x)
	_, y2 := getPoint(state, prob, l.P2ID, x)
	dy := y1 - y2
	return dy * dy
}

func evalVertical(state *SketchState, prob *problemState, x []float64, c Constraint) float64 {
	if len(c.EntityIDs) != 1 {
		return 0
	}
	l := getLine(state, c.EntityIDs[0])
	if l == nil {
		return 0
	}
	x1, _ := getPoint(state, prob, l.P1ID, x)
	x2, _ := getPoint(state, prob, l.P2ID, x)
	dx := x1 - x2
	return dx * dx
}

func evalParallel(state *SketchState, prob *problemState, x []float64, c Constraint) float64 {
	if len(c.EntityIDs) != 2 {
		return 0
	}
	l1 := getLine(state, c.EntityIDs[0])
	l2 := getLine(state, c.EntityIDs[1])
	if l1 == nil || l2 == nil {
		return 0
	}
	x1, y1 := getPoint(state, prob, l1.P1ID, x)
	x2, y2 := getPoint(state, prob, l1.P2ID, x)
	x3, y3 := getPoint(state, prob, l2.P1ID, x)
	x4, y4 := getPoint(state, prob, l2.P2ID, x)
	crossProduct := (x2-x1)*(y4-y3) - (y2-y1)*(x4-x3)
	return crossProduct * crossProduct
}

func evalPerpendicular(state *SketchState, prob *problemState, x []float64, c Constraint) float64 {
	if len(c.EntityIDs) != 2 {
		return 0
	}
	l1 := getLine(state, c.EntityIDs[0])
	l2 := getLine(state, c.EntityIDs[1])
	if l1 == nil || l2 == nil {
		return 0
	}
	x1, y1 := getPoint(state, prob, l1.P1ID, x)
	x2, y2 := getPoint(state, prob, l1.P2ID, x)
	x3, y3 := getPoint(state, prob, l2.P1ID, x)
	x4, y4 := getPoint(state, prob, l2.P2ID, x)
	dotProduct := (x2-x1)*(x4-x3) + (y2-y1)*(y4-y3)
	return dotProduct * dotProduct
}

func evalPointLineDistance(state *SketchState, prob *problemState, x []float64, c Constraint) float64 {
	if len(c.EntityIDs) != 2 {
		return 0
	}
	px, py := getPoint(state, prob, c.EntityIDs[0], x)
	l := getLine(state, c.EntityIDs[1])
	if l == nil {
		return 0
	}
	x1, y1 := getPoint(state, prob, l.P1ID, x)
	x2, y2 := getPoint(state, prob, l.P2ID, x)
	dx, dy := x2-x1, y2-y1
	den := math.Sqrt(dx*dx + dy*dy)
	if den == 0 {
		return 0
	}
	dist := math.Abs(dy*px-dx*py+x2*y1-y2*x1) / den
	err := dist - c.Value
	return err * err
}
