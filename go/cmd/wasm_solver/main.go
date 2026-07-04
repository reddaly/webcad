// Package main provides the WebAssembly entry point for the Geometric Constraint Solver.
package main

import (
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
		return solver.EncodeError(err.Error())
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
	state, err := solver.DecodeSketchState(inputJSON)
	if err != nil {
		return solver.EncodeError(fmt.Sprintf("Invalid input JSON: %v", err))
	}

	result := solver.Solve(state, algo)

	output, err := solver.EncodeResult(result)
	if err != nil {
		return solver.EncodeError(fmt.Sprintf("Failed to serialize result: %v", err))
	}

	return output
}
