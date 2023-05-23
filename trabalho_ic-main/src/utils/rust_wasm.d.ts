/* tslint:disable */
/* eslint-disable */
/**
* @param {string} data_str
* @param {number} pop_len
* @param {number} n_params
* @param {number} answ_index
* @param {number} n_max_iter
* @param {number} step_size
* @returns {Wrapper}
*/
export function new_wrapper_from_string(data_str: string, pop_len: number, n_params: number, answ_index: number, n_max_iter: number, step_size: number): Wrapper;
/**
*/
export class Wrapper {
  free(): void;
/**
* @param {number} n_ger
* @param {number} mutation_rate
* @param {number} crosssover_rate
* @returns {any[]}
*/
  train_wrapper(n_ger: number, mutation_rate: number, crosssover_rate: number): any[];
/**
* @param {number} mutation_rate
* @param {number} crossover_rate
*/
  update_wrapper(mutation_rate: number, crossover_rate: number): void;
/**
* @returns {string}
*/
  get_wrapper(): string;
}

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
  readonly memory: WebAssembly.Memory;
  readonly __wbg_wrapper_free: (a: number) => void;
  readonly wrapper_train_wrapper: (a: number, b: number, c: number, d: number, e: number) => void;
  readonly wrapper_update_wrapper: (a: number, b: number, c: number) => void;
  readonly wrapper_get_wrapper: (a: number, b: number) => void;
  readonly new_wrapper_from_string: (a: number, b: number, c: number, d: number, e: number, f: number, g: number) => number;
  readonly __wbindgen_malloc: (a: number) => number;
  readonly __wbindgen_realloc: (a: number, b: number, c: number) => number;
  readonly __wbindgen_add_to_stack_pointer: (a: number) => number;
  readonly __wbindgen_free: (a: number, b: number) => void;
  readonly __wbindgen_exn_store: (a: number) => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;
/**
* Instantiates the given `module`, which can either be bytes or
* a precompiled `WebAssembly.Module`.
*
* @param {SyncInitInput} module
*
* @returns {InitOutput}
*/
export function initSync(module: SyncInitInput): InitOutput;

/**
* If `module_or_path` is {RequestInfo} or {URL}, makes a request and
* for everything else, calls `WebAssembly.instantiate` directly.
*
* @param {InitInput | Promise<InitInput>} module_or_path
*
* @returns {Promise<InitOutput>}
*/
export default function __wbg_init (module_or_path?: InitInput | Promise<InitInput>): Promise<InitOutput>;
