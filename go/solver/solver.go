package solver

import (
	"math"

	"gonum.org/v1/gonum/optimize"
)

// SolverAlgorithm defines the algorithm used for optimization.
type SolverAlgorithm string

const (
	AlgorithmBFGS SolverAlgorithm = "bfgs"
	AlgorithmLM   SolverAlgorithm = "lm" // Levenberg-Marquardt (approximated or via specific gonum solver if available)
)

// Solve resolves the sketch constraints to find the optimal point positions.
func Solve(state SketchState, algo SolverAlgorithm) *SolverResult {
	// First, gather all variables we need to optimize (unfixed points' X and Y).
	// We'll maintain a mapping from point ID to the index in our optimization slice.
	var initialX []float64
	pointIdx := make(map[string]int)
	
	for _, p := range state.Points {
		if !p.Fixed {
			pointIdx[p.ID] = len(initialX)
			initialX = append(initialX, p.X, p.Y)
		}
	}

	// For circles with unfixed radii, we might also want to optimize them,
	// but to keep things simple for now we'll stick to points. 
	// Wait, standard GCS usually optimizes radii too if not fixed.
	circleIdx := make(map[string]int)
	for _, c := range state.Circles {
		if !c.FixedRadius {
			circleIdx[c.ID] = len(initialX)
			initialX = append(initialX, c.Radius)
		}
	}

	// Helper to get point coordinates
	getPoint := func(id string, x []float64) (float64, float64) {
		if idx, ok := pointIdx[id]; ok {
			return x[idx], x[idx+1]
		}
		for _, p := range state.Points {
			if p.ID == id {
				return p.X, p.Y
			}
		}
		return 0, 0
	}

	// Helper to get line
	getLine := func(id string) *Line {
		for _, l := range state.Lines {
			if l.ID == id {
				return &l
			}
		}
		return nil
	}

	// Objective function: sum of squared errors
	objFunc := func(x []float64) float64 {
		var totalError float64

		for _, c := range state.Constraints {
			switch c.Type {
			case "coincident":
				if len(c.EntityIDs) == 2 {
					x1, y1 := getPoint(c.EntityIDs[0], x)
					x2, y2 := getPoint(c.EntityIDs[1], x)
					dx, dy := x1-x2, y1-y2
					totalError += dx*dx + dy*dy
				}
			case "distance":
				if len(c.EntityIDs) == 2 {
					x1, y1 := getPoint(c.EntityIDs[0], x)
					x2, y2 := getPoint(c.EntityIDs[1], x)
					dx, dy := x1-x2, y1-y2
					dist := math.Sqrt(dx*dx + dy*dy)
					err := dist - c.Value
					totalError += err * err
				}
			case "horizontalDistance":
				if len(c.EntityIDs) == 2 {
					x1, _ := getPoint(c.EntityIDs[0], x)
					x2, _ := getPoint(c.EntityIDs[1], x)
					dx := x1 - x2
					err := math.Abs(dx) - c.Value
					totalError += err * err
				}
			case "verticalDistance":
				if len(c.EntityIDs) == 2 {
					_, y1 := getPoint(c.EntityIDs[0], x)
					_, y2 := getPoint(c.EntityIDs[1], x)
					dy := y1 - y2
					err := math.Abs(dy) - c.Value
					totalError += err * err
				}
			case "horizontal":
				if len(c.EntityIDs) == 1 {
					if l := getLine(c.EntityIDs[0]); l != nil {
						_, y1 := getPoint(l.P1ID, x)
						_, y2 := getPoint(l.P2ID, x)
						dy := y1 - y2
						totalError += dy * dy
					}
				}
			case "vertical":
				if len(c.EntityIDs) == 1 {
					if l := getLine(c.EntityIDs[0]); l != nil {
						x1, _ := getPoint(l.P1ID, x)
						x2, _ := getPoint(l.P2ID, x)
						dx := x1 - x2
						totalError += dx * dx
					}
				}
			case "parallel":
				if len(c.EntityIDs) == 2 {
					l1 := getLine(c.EntityIDs[0])
					l2 := getLine(c.EntityIDs[1])
					if l1 != nil && l2 != nil {
						x1, y1 := getPoint(l1.P1ID, x)
						x2, y2 := getPoint(l1.P2ID, x)
						x3, y3 := getPoint(l2.P1ID, x)
						x4, y4 := getPoint(l2.P2ID, x)
						dx1, dy1 := x2-x1, y2-y1
						dx2, dy2 := x4-x3, y4-y3
						crossProduct := dx1*dy2 - dy1*dx2
						totalError += crossProduct * crossProduct
					}
				}
			case "perpendicular":
				if len(c.EntityIDs) == 2 {
					l1 := getLine(c.EntityIDs[0])
					l2 := getLine(c.EntityIDs[1])
					if l1 != nil && l2 != nil {
						x1, y1 := getPoint(l1.P1ID, x)
						x2, y2 := getPoint(l1.P2ID, x)
						x3, y3 := getPoint(l2.P1ID, x)
						x4, y4 := getPoint(l2.P2ID, x)
						dx1, dy1 := x2-x1, y2-y1
						dx2, dy2 := x4-x3, y4-y3
						dotProduct := dx1*dx2 + dy1*dy2
						totalError += dotProduct * dotProduct
					}
				}
			case "pointLineDistance":
				if len(c.EntityIDs) == 2 {
					px, py := getPoint(c.EntityIDs[0], x)
					l := getLine(c.EntityIDs[1])
					if l != nil {
						x1, y1 := getPoint(l.P1ID, x)
						x2, y2 := getPoint(l.P2ID, x)
						dx, dy := x2-x1, y2-y1
						num := math.Abs(dy*px - dx*py + x2*y1 - y2*x1)
						den := math.Sqrt(dx*dx + dy*dy)
						if den != 0 {
							dist := num / den
							err := dist - c.Value
							totalError += err * err
						}
					}
				}
			}
		}

		return totalError
	}

	problem := optimize.Problem{
		Func: objFunc,
	}

	var method optimize.Method
	if algo == AlgorithmLM {
		// Gonum's optimize package might not have LM built-in natively, or we can map it to CG/BFGS for now
		// Actually, LM is for least squares, but since we are formulating SSE directly, we can just use BFGS or CG.
		// We'll use BFGS as a fallback if LM is requested but not available.
		method = &optimize.BFGS{}
	} else {
		method = &optimize.BFGS{}
	}

	var result *optimize.Result
	var err error

	if len(initialX) > 0 {
		settings := &optimize.Settings{
			GradientThreshold: 1e-6,
		}
		result, err = optimize.Minimize(problem, initialX, settings, method)
		if err != nil {
			return &SolverResult{
				Success: false,
				Error:   err.Error(),
			}
		}
		initialX = result.X
	}

	// Reconstruct the solved state
	res := &SolverResult{
		Success: true,
	}
	for _, p := range state.Points {
		if idx, ok := pointIdx[p.ID]; ok {
			p.X = initialX[idx]
			p.Y = initialX[idx+1]
		}
		res.Points = append(res.Points, p)
	}

	for _, c := range state.Circles {
		if idx, ok := circleIdx[c.ID]; ok {
			c.Radius = initialX[idx]
		}
		res.Circles = append(res.Circles, c)
	}

	return res
}
