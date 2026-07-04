package main

import (
	"encoding/json"
	"fmt"
	"syscall/js"

	"github.com/gonzojive/webcad/go/solver"
)

func main() {
	c := make(chan struct{}, 0)
	js.Global().Set("solve_gcs", js.FuncOf(solveGCS))
	fmt.Println("Go WebAssembly initialized")
	<-c
}

func solveGCS(this js.Value, args []js.Value) interface{} {
	if len(args) < 1 {
		return errorJSON("Missing arguments")
	}

	inputJSON := args[0].String()

	// Parse optional algorithm parameter, default to BFGS
	algo := solver.AlgorithmBFGS
	if len(args) > 1 {
		algoStr := args[1].String()
		if algoStr == string(solver.AlgorithmLM) {
			algo = solver.AlgorithmLM
		}
	}

	var state solver.SketchState
	if err := json.Unmarshal([]byte(inputJSON), &state); err != nil {
		return errorJSON("Invalid input JSON: " + err.Error())
	}

	result := solver.Solve(state, algo)
	
	output, err := json.Marshal(result)
	if err != nil {
		return errorJSON("Failed to serialize result: " + err.Error())
	}

	return string(output)
}

func errorJSON(msg string) string {
	res := solver.SolverResult{
		Success: false,
		Error:   msg,
	}
	b, _ := json.Marshal(res)
	return string(b)
}
