// Package main provides the WebAssembly entry point for the Geometric Constraint Solver.
package main

import (
	"encoding/json"
	"fmt"
	"syscall/js"

	"github.com/gonzojive/webcad/go/solvers/bfgs"
	"github.com/gonzojive/webcad/go/solvers/lm"
	"github.com/gonzojive/webcad/proto"
	"google.golang.org/protobuf/encoding/protojson"
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
		return encodeError(err.Error())
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
// arguments if provided, defaulting to the LM algorithm otherwise.
func parseAlgorithm(args []js.Value) string {
	if len(args) > 1 {
		return args[1].String()
	}
	// The robust Levenberg-Marquardt solver is the default.
	return "lm"
}

// processSolveRequest deserializes the JSON sketch state, invokes the core
// Go geometric constraint solver, and serializes the result back to JSON.
func processSolveRequest(inputJSON string, algo string) string {
	var sketch schema.Sketch
	
	unmarshaler := protojson.UnmarshalOptions{
		DiscardUnknown: true,
	}
	if err := unmarshaler.Unmarshal([]byte(inputJSON), &sketch); err != nil {
		return encodeError(fmt.Sprintf("Invalid input JSON: %v", err))
	}

	var result *schema.SolveResult
	var err error

	if algo == "bfgs" {
		solver := bfgs.New()
		result, err = solver.Solve(&sketch)
	} else {
		solver := lm.New()
		result, err = solver.Solve(&sketch)
	}

	if err != nil {
		return encodeError(fmt.Sprintf("Solver error: %v", err))
	}

	marshaler := protojson.MarshalOptions{
		EmitUnpopulated: true,
	}
	
	output, err := marshaler.Marshal(result)
	if err != nil {
		return encodeError(fmt.Sprintf("Failed to serialize result: %v", err))
	}

	return string(output)
}

// encodeError creates a JSON error response string.
func encodeError(msg string) string {
	errResp := map[string]interface{}{
		"success":       false,
		"error_message": msg,
	}
	b, _ := json.Marshal(errResp)
	return string(b)
}
