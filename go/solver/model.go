package solver

type EntityType string

const (
	PointType  EntityType = "point"
	LineType   EntityType = "line"
	CircleType EntityType = "circle"
)

type Point struct {
	ID    string  `json:"id"`
	X     float64 `json:"x"`
	Y     float64 `json:"y"`
	Fixed bool    `json:"fixed,omitempty"`
}

type Line struct {
	ID   string `json:"id"`
	P1ID string `json:"p1Id"`
	P2ID string `json:"p2Id"`
}

type Circle struct {
	ID          string  `json:"id"`
	CenterID    string  `json:"centerId"`
	Radius      float64 `json:"radius"`
	FixedRadius bool    `json:"fixedRadius,omitempty"`
}

type Constraint struct {
	ID        string   `json:"id"`
	Type      string   `json:"type"`
	EntityIDs []string `json:"entityIds"`
	Value     float64  `json:"value,omitempty"`
}

type SketchState struct {
	Points      []Point      `json:"points"`
	Lines       []Line       `json:"lines"`
	Circles     []Circle     `json:"circles"`
	Constraints []Constraint `json:"constraints"`
}

type SolverResult struct {
	Success bool     `json:"success"`
	Points  []Point  `json:"points"`
	Circles []Circle `json:"circles"`
	Error   string   `json:"error,omitempty"`
}
