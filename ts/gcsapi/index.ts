/**
 * GCS API
 * A high-quality TypeScript library for working with geometric constraint solvers.
 */

// Model Types

export type EntityType = 'point' | 'line' | 'circle';

export interface GCSPoint {
    id: string; // Unique string identifier
    x: number;
    y: number;
    fixed?: boolean; // Lock X and Y coordinates in GCS solver
}

export interface GCSLine {
    id: string;
    p1Id: string; // Start point entity ID
    p2Id: string; // End point entity ID
}

export interface GCSCircle {
    id: string;
    centerId: string; // Center point entity ID
    radius: number; // Current radius value
    fixedRadius?: boolean; // Lock radius length in GCS solver
}

// Constraint Types
export interface BaseConstraint {
    id: string;
    type: string;
}

export interface CoincidentConstraint extends BaseConstraint {
    type: 'coincident';
    p1Id: string;
    p2Id: string;
}

export interface DistanceConstraint extends BaseConstraint {
    type: 'distance';
    p1Id: string;
    p2Id: string;
    value: number;
}

export interface HorizontalDistanceConstraint extends BaseConstraint {
    type: 'horizontalDistance';
    p1Id: string;
    p2Id: string;
    value: number;
}

export interface VerticalDistanceConstraint extends BaseConstraint {
    type: 'verticalDistance';
    p1Id: string;
    p2Id: string;
    value: number;
}

export interface PointLineDistanceConstraint extends BaseConstraint {
    type: 'pointLineDistance';
    pointId: string;
    lineId: string;
    value: number;
}

export interface VerticalConstraint extends BaseConstraint {
    type: 'vertical';
    lineId: string;
}

export interface HorizontalConstraint extends BaseConstraint {
    type: 'horizontal';
    lineId: string;
}

export interface ParallelConstraint extends BaseConstraint {
    type: 'parallel';
    line1Id: string;
    line2Id: string;
}

export interface PerpendicularConstraint extends BaseConstraint {
    type: 'perpendicular';
    line1Id: string;
    line2Id: string;
}

export type GCSConstraint =
    | CoincidentConstraint
    | DistanceConstraint
    | HorizontalDistanceConstraint
    | VerticalDistanceConstraint
    | PointLineDistanceConstraint
    | VerticalConstraint
    | HorizontalConstraint
    | ParallelConstraint
    | PerpendicularConstraint;

export interface GCSSketchState {
    points: GCSPoint[];
    lines: GCSLine[];
    circles: GCSCircle[];
    constraints: GCSConstraint[];
}

export interface SolverResult {
    success: boolean;
    points: GCSPoint[];
    circles: GCSCircle[];
    error?: string;
}

// Internal structures expected by our Go WASM solver
interface GoConstraint {
    id: string;
    type: string;
    entityIds: string[];
    value?: number;
}

interface GoSketchState {
    points: GCSPoint[];
    lines: GCSLine[];
    circles: GCSCircle[];
    constraints: GoConstraint[];
}

/**
 * The available solver algorithms.
 */
export type SolverAlgorithm = 'bfgs' | 'lm' | 'ezpz';

export interface SolverOptions {
    algorithm?: SolverAlgorithm;
}

/**
 * Global interface for our Go-injected function
 */
declare global {
    function solve_gcs(inputJson: string, algo: string): string;
}

/**
 * Geometric Constraint Solver class providing a clean API.
 */
export class GCSSolver {
    private goWasmInitialized = false;

    /**
     * Initializes the solver WASM. 
     * In a real web environment, this will load the go_wasm_exec.js and instantiate the module.
     */
    async initGoWasm(wasmUrl: string): Promise<void> {
        if (this.goWasmInitialized) return;
        
        // This assumes that the Go standard lib `wasm_exec.js` is already loaded 
        // into the global scope (providing the Go class).
        if (typeof (globalThis as any).Go === 'undefined') {
            throw new Error("Go wasm_exec.js is not loaded in the global scope.");
        }

        const go = new (globalThis as any).Go();
        
        // Use fetch or streaming instantiate
        if (typeof WebAssembly.instantiateStreaming === "function") {
            const obj = await WebAssembly.instantiateStreaming(fetch(wasmUrl), go.importObject);
            go.run(obj.instance);
        } else {
            const resp = await fetch(wasmUrl);
            const bytes = await resp.arrayBuffer();
            const obj = await WebAssembly.instantiate(bytes, go.importObject);
            go.run(obj.instance);
        }
        
        this.goWasmInitialized = true;
    }

    /**
     * Solves the given sketch state using the specified algorithm.
     */
    solve(state: GCSSketchState, options?: SolverOptions): SolverResult {
        const algo = options?.algorithm || 'bfgs';
        
        // We can add integration for 'ezpz' Rust solver if needed, 
        // but this lib specifically supports our Go solvers 'lm' and 'bfgs'.
        if (algo === 'bfgs' || algo === 'lm') {
            return this.solveWithGo(state, algo);
        } else {
            throw new Error(`Algorithm ${algo} is not supported directly in this solver module. Use ezpz wrapper instead.`);
        }
    }

    private solveWithGo(state: GCSSketchState, algo: 'bfgs' | 'lm'): SolverResult {
        if (!this.goWasmInitialized) {
            throw new Error("Go WASM solver is not initialized. Call initGoWasm() first.");
        }

        if (typeof solve_gcs === 'undefined') {
            throw new Error("solve_gcs function not found. Did the Go WASM module initialize correctly?");
        }

        // Convert the highly-typed TS constraints into the flat JSON structure expected by the Go solver
        const goConstraints: GoConstraint[] = state.constraints.map(c => {
            switch (c.type) {
                case 'coincident':
                case 'distance':
                case 'horizontalDistance':
                case 'verticalDistance':
                    return {
                        id: c.id,
                        type: c.type,
                        entityIds: [c.p1Id, c.p2Id],
                        value: 'value' in c ? c.value : undefined
                    };
                case 'pointLineDistance':
                    return {
                        id: c.id,
                        type: c.type,
                        entityIds: [c.pointId, c.lineId],
                        value: c.value
                    };
                case 'vertical':
                case 'horizontal':
                    return {
                        id: c.id,
                        type: c.type,
                        entityIds: [c.lineId]
                    };
                case 'parallel':
                case 'perpendicular':
                    return {
                        id: c.id,
                        type: c.type,
                        entityIds: [c.line1Id, c.line2Id]
                    };
                default:
                    throw new Error(`Unsupported constraint type: ${(c as any).type}`);
            }
        });

        const goState: GoSketchState = {
            points: state.points,
            lines: state.lines,
            circles: state.circles,
            constraints: goConstraints
        };

        const inputJson = JSON.stringify(goState);
        
        try {
            // Call the Go WASM function
            const outputJson = solve_gcs(inputJson, algo);
            return JSON.parse(outputJson) as SolverResult;
        } catch (e) {
            return {
                success: false,
                points: [],
                circles: [],
                error: e instanceof Error ? e.message : String(e)
            };
        }
    }
}
