// Package main provides the WebAssembly entry point for the Geometric Constraint Solver.
package main

import (
	"encoding/json"
	"fmt"
	"syscall/js"

	"github.com/gonzojive/webcad/go/solver"
)

func main() {
	c := make(chan struct{})
	js.Global().Set("solve_gcs", js.FuncOf(solveGCS))
	fmt.Println("Go WebAssembly initialized")
	<-c
}

// solveGCS is the exported WASM function that receives a JSON sketch state
// and an algorithm, minimizes the geometric error, and returns the solved JSON state.
func solveGCS(this js.Value, args []js.Value) interface{} {
	if err := validateArgs(args); err != nil {
		return errorJSON(err.Error())
	}

	inputJSON := args[0].String()
	algo := parseAlgorithm(args)

	return processSolveRequest(inputJSON, algo)
}

// validateArgs ensures that the WASM function received the minimum required
// number of arguments from the JavaScript environment.
func validateArgs(args []js.Value) error {
	if len(args) < 1 {
		return fmt.Errorf("missing arguments: requires at least the JSON sketch state")
	}
	return nil
}

// parseAlgorithm extracts the requested solver algorithm from the JavaScript
// arguments if provided, defaulting to the BFGS algorithm otherwise.
func parseAlgorithm(args []js.Value) solver.SolverAlgorithm {
	if len(args) > 1 && args[1].String() == string(solver.AlgorithmLM) {
		return solver.AlgorithmLM
	}
	return solver.AlgorithmBFGS
}

// processSolveRequest deserializes the JSON sketch state, invokes the core
// Go geometric constraint solver, and serializes the result back to JSON.
func processSolveRequest(inputJSON string, algo solver.SolverAlgorithm) string {
	var state solver.SketchState
	if err := json.Unmarshal([]byte(inputJSON), &state); err != nil {
		return errorJSON(fmt.Sprintf("Invalid input JSON: %v", err))
	}

	result := solver.Solve(state, algo)
	
	output, err := json.Marshal(result)
	if err != nil {
		return errorJSON(fmt.Sprintf("Failed to serialize result: %v", err))
	}

	return string(output)
}

// errorJSON is a helper that generates a serialized JSON SolverResult
// indicating a failure, encapsulating the provided error message.
func errorJSON(msg string) string {
	res := solver.SolverResult{
		Success: false,
		Error:   msg,
	}
	b, _ := json.Marshal(res)
	return string(b)
}
