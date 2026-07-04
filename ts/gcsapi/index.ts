/**
 * GCS API
 * A high-quality TypeScript library for working with geometric constraint solvers.
 */

/** 
 * EntityId uniquely identifies a geometric entity in the sketch. 
 */
export type EntityId = string;

/** 
 * EntityType specifies the category of a geometric entity. 
 */
export type EntityType = 'point' | 'line' | 'circle';

/**
 * GCSPoint represents a 2D coordinate in the geometric sketch.
 */
export interface GCSPoint {
    id: EntityId;
    x: number;
    y: number;
    fixed?: boolean; // Lock X and Y coordinates in GCS solver
}

/**
 * GCSLine represents a line segment connecting two points.
 */
export interface GCSLine {
    id: EntityId;
    p1Id: EntityId; // Start point entity ID
    p2Id: EntityId; // End point entity ID
}

/**
 * GCSCircle represents a circle defined by a center point and a radius.
 */
export interface GCSCircle {
    id: EntityId;
    centerId: EntityId; // Center point entity ID
    radius: number; // Current radius value
    fixedRadius?: boolean; // Lock radius length in GCS solver
}

/**
 * BaseConstraint provides the foundational fields for any geometric constraint.
 */
export interface BaseConstraint {
    id: EntityId;
    type: string;
}

/**
 * CoincidentConstraint forces two points to share the same coordinates.
 */
export interface CoincidentConstraint extends BaseConstraint {
    type: 'coincident';
    p1Id: EntityId;
    p2Id: EntityId;
}

/**
 * DistanceConstraint forces the distance between two points to equal a specific value.
 */
export interface DistanceConstraint extends BaseConstraint {
    type: 'distance';
    p1Id: EntityId;
    p2Id: EntityId;
    value: number;
}

/**
 * HorizontalDistanceConstraint forces the horizontal distance (delta X) between two points to equal a specific value.
 */
export interface HorizontalDistanceConstraint extends BaseConstraint {
    type: 'horizontalDistance';
    p1Id: EntityId;
    p2Id: EntityId;
    value: number;
}

/**
 * VerticalDistanceConstraint forces the vertical distance (delta Y) between two points to equal a specific value.
 */
export interface VerticalDistanceConstraint extends BaseConstraint {
    type: 'verticalDistance';
    p1Id: EntityId;
    p2Id: EntityId;
    value: number;
}

/**
 * PointLineDistanceConstraint forces the perpendicular distance from a point to a line to equal a specific value.
 */
export interface PointLineDistanceConstraint extends BaseConstraint {
    type: 'pointLineDistance';
    pointId: EntityId;
    lineId: EntityId;
    value: number;
}

/**
 * VerticalConstraint forces a line to be perfectly vertical.
 */
export interface VerticalConstraint extends BaseConstraint {
    type: 'vertical';
    lineId: EntityId;
}

/**
 * HorizontalConstraint forces a line to be perfectly horizontal.
 */
export interface HorizontalConstraint extends BaseConstraint {
    type: 'horizontal';
    lineId: EntityId;
}

/**
 * ParallelConstraint forces two lines to be parallel to each other.
 */
export interface ParallelConstraint extends BaseConstraint {
    type: 'parallel';
    line1Id: EntityId;
    line2Id: EntityId;
}

/**
 * PerpendicularConstraint forces two lines to be perpendicular to each other.
 */
export interface PerpendicularConstraint extends BaseConstraint {
    type: 'perpendicular';
    line1Id: EntityId;
    line2Id: EntityId;
}

/**
 * GCSConstraint is a union of all possible geometric constraint types.
 */
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

/**
 * GCSSketchState holds the entire state of a geometric sketch prior to or after solving.
 */
export interface GCSSketchState {
    points: GCSPoint[];
    lines: GCSLine[];
    circles: GCSCircle[];
    constraints: GCSConstraint[];
}

/**
 * SolverResult represents the final output of the constraint solver.
 */
export interface SolverResult {
    success: boolean;
    points: GCSPoint[];
    circles: GCSCircle[];
    error?: string;
}

// Internal structures expected by our Go WASM solver
interface GoConstraint {
    id: EntityId;
    type: string;
    entityIds: EntityId[];
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

/**
 * SolverOptions configures the geometric solver execution.
 */
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
     * 
     * @param wasmUrl The URL to fetch the compiled WebAssembly module from.
     */
    async initGoWasm(wasmUrl: string): Promise<void> {
        if (this.goWasmInitialized) return;
        
        if (typeof (globalThis as any).Go === 'undefined') {
            throw new Error("Go wasm_exec.js is not loaded in the global scope.");
        }

        const go = new (globalThis as any).Go();
        
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
     * 
     * @param state The sketch layout and constraints to solve.
     * @param options Execution configuration.
     * @returns A SolverResult detailing the success or failure and the updated geometry.
     */
    solve(state: GCSSketchState, options?: SolverOptions): SolverResult {
        const algo = options?.algorithm || 'bfgs';
        
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

        const goConstraints = this.mapConstraintsToGo(state.constraints);

        const goState: GoSketchState = {
            points: state.points,
            lines: state.lines,
            circles: state.circles,
            constraints: goConstraints
        };

        const inputJson = JSON.stringify(goState);
        
        try {
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

    private mapConstraintsToGo(constraints: GCSConstraint[]): GoConstraint[] {
        return constraints.map(c => {
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
    }
}
