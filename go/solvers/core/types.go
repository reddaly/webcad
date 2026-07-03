// Package core defines the primary interfaces and types used across the webcad
// solver library.
package core

// SketchID uniquely identifies a sketch.
// It is typically used to correlate results back to the source sketch.
type SketchID string

// EntityID uniquely identifies a geometric entity (e.g., point, line, circle)
// within a sketch.
type EntityID string

// SolverID uniquely identifies a solver implementation.
// This ID is used to identify which solver produced the results.
type SolverID string

