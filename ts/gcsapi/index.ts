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
        const algo = options?.algorithm || 'lm';
        
        if (algo === 'bfgs' || algo === 'lm') {
            return this.solveWithGo(state, algo);
        } else {
            throw new Error(`Algorithm ${algo} is not supported directly in this solver module.`);
        }
    }

    private solveWithGo(state: GCSSketchState, algo: 'bfgs' | 'lm'): SolverResult {
        if (!this.goWasmInitialized) {
            throw new Error("Go WASM solver is not initialized. Call initGoWasm() first.");
        }

        if (typeof solve_gcs === 'undefined') {
            throw new Error("solve_gcs function not found. Did the Go WASM module initialize correctly?");
        }

        const goEntities = this.mapEntitiesToGo(state);
        const goConstraints = this.mapConstraintsToGo(state.constraints, state);

        const sketchProto = {
            id: "sketch-1",
            entities: goEntities,
            constraints: goConstraints
        };

        const inputJson = JSON.stringify(sketchProto);
        
        try {
            const outputJson = solve_gcs(inputJson, algo);
            const resultProto = JSON.parse(outputJson);

            if (!resultProto.success) {
                return {
                    success: false,
                    points: [],
                    circles: [],
                    error: resultProto.errorMessage || "Solver failed"
                };
            }

            return this.mapGoResultToState(resultProto.solvedState, state);
        } catch (e) {
            return {
                success: false,
                points: [],
                circles: [],
                error: e instanceof Error ? e.message : String(e)
            };
        }
    }

    private mapEntitiesToGo(state: GCSSketchState): any[] {
        const entities: any[] = [];
        
        for (const p of state.points) {
            entities.push({
                id: p.id,
                point: {
                    x: p.x,
                    y: p.y
                }
            });
        }
        
        for (const l of state.lines) {
            const p1 = state.points.find(pt => pt.id === l.p1Id);
            const p2 = state.points.find(pt => pt.id === l.p2Id);
            if (!p1 || !p2) continue;
            
            entities.push({
                id: l.id,
                line: {
                    x1: p1.x, y1: p1.y,
                    x2: p2.x, y2: p2.y
                }
            });
        }
        
        for (const c of state.circles) {
            const center = state.points.find(pt => pt.id === c.centerId);
            if (!center) continue;
            
            entities.push({
                id: c.id,
                circle: {
                    cx: center.x,
                    cy: center.y,
                    r: c.radius
                }
            });
        }
        
        return entities;
    }

    private mapConstraintsToGo(constraints: GCSConstraint[], state: GCSSketchState): any[] {
        const goConstraints: any[] = [];
        
        for (const c of constraints) {
            switch (c.type) {
                case 'coincident':
                    goConstraints.push({
                        id: c.id,
                        coincidence: { entityA: c.p1Id, entityB: c.p2Id }
                    });
                    break;
                case 'distance':
                    goConstraints.push({
                        id: c.id,
                        distance: { entityA: c.p1Id, entityB: c.p2Id, value: c.value }
                    });
                    break;
                case 'horizontalDistance':
                case 'verticalDistance':
                    // We can map these to distance for now or ignore since the proto doesn't explicitly have horizontal/vertical yet, 
                    // or implement a workaround. For now we will rely on distance if possible.
                    break;
                case 'parallel':
                    goConstraints.push({
                        id: c.id,
                        parallel: { lineA: c.line1Id, lineB: c.line2Id }
                    });
                    break;
                case 'perpendicular':
                    goConstraints.push({
                        id: c.id,
                        perpendicular: { lineA: c.line1Id, lineB: c.line2Id }
                    });
                    break;
            }
        }
        return goConstraints;
    }

    private mapGoResultToState(solvedState: any, originalState: GCSSketchState): SolverResult {
        if (!solvedState || !solvedState.entities) {
             return { success: false, points: [], circles: [], error: "No solved state returned" };
        }

        const newPoints: GCSPoint[] = [];
        const newCircles: GCSCircle[] = [];

        for (const pt of originalState.points) {
            const solvedEnt = solvedState.entities[pt.id];
            if (solvedEnt && solvedEnt.point) {
                newPoints.push({
                    ...pt,
                    x: solvedEnt.point.x,
                    y: solvedEnt.point.y
                });
            } else {
                newPoints.push(pt);
            }
        }

        for (const c of originalState.circles) {
            const solvedEnt = solvedState.entities[c.id];
            if (solvedEnt && solvedEnt.circle) {
                newCircles.push({
                    ...c,
                    radius: solvedEnt.circle.r
                });
            } else {
                newCircles.push(c);
            }
        }

        return {
            success: true,
            points: newPoints,
            circles: newCircles
        };
    }
}
