// Package solver provides a geometric constraint solving engine.
package solver

// EntityID represents a unique identifier for any geometric entity in the sketch.
type EntityID string

// EntityType represents the category of a geometric entity.
type EntityType string

const (
	// PointType represents a 2D point entity.
	PointType EntityType = "point"
	// LineType represents a line segment defined by two points.
	LineType EntityType = "line"
	// CircleType represents a circle defined by a center point and a radius.
	CircleType EntityType = "circle"
)

// ConstraintType represents the specific type of geometric constraint.
type ConstraintType string

const (
	ConstraintCoincident         ConstraintType = "coincident"
	ConstraintDistance           ConstraintType = "distance"
	ConstraintHorizontalDistance ConstraintType = "horizontalDistance"
	ConstraintVerticalDistance   ConstraintType = "verticalDistance"
	ConstraintPointLineDistance  ConstraintType = "pointLineDistance"
	ConstraintHorizontal         ConstraintType = "horizontal"
	ConstraintVertical           ConstraintType = "vertical"
	ConstraintParallel           ConstraintType = "parallel"
	ConstraintPerpendicular      ConstraintType = "perpendicular"
)

// Point defines a 2D coordinate in the geometric sketch.
type Point struct {
	ID    EntityID `json:"id"`
	X     float64  `json:"x"`
	Y     float64  `json:"y"`
	Fixed bool     `json:"fixed,omitempty"`
}

// Line defines a line segment connecting two points.
type Line struct {
	ID   EntityID `json:"id"`
	P1ID EntityID `json:"p1Id"`
	P2ID EntityID `json:"p2Id"`
}

// Circle defines a circle with a center point and a radius.
type Circle struct {
	ID          EntityID `json:"id"`
	CenterID    EntityID `json:"centerId"`
	Radius      float64  `json:"radius"`
	FixedRadius bool     `json:"fixedRadius,omitempty"`
}

// Constraint defines a geometric rule applied to one or more entities.
type Constraint struct {
	ID        EntityID       `json:"id"`
	Type      ConstraintType `json:"type"`
	EntityIDs []EntityID     `json:"entityIds"`
	Value     float64        `json:"value,omitempty"`
}

// SketchState represents the complete state of a geometric sketch.
type SketchState struct {
	Points      []Point      `json:"points"`
	Lines       []Line       `json:"lines"`
	Circles     []Circle     `json:"circles"`
	Constraints []Constraint `json:"constraints"`
}

// SolverResult contains the output of the constraint solving process.
type SolverResult struct {
	Success bool     `json:"success"`
	Points  []Point  `json:"points"`
	Circles []Circle `json:"circles"`
	Error   string   `json:"error,omitempty"`
}
