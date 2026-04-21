declare module '@tweakpane/core' {
  export interface Semver { major: number; minor: number; patch: number; }
  export interface BaseParams { disabled?: boolean; hidden?: boolean; }
  export interface BaseBladeParams extends BaseParams { view: string; }
  export type BindingParams = Record<string, unknown>;
  export interface BindingApiEvents<T> { change: TpChangeEvent<T>; }
  export interface BladeApi<C = unknown> { readonly controller: C; dispose(): void; hidden: boolean; disabled: boolean; }
  export interface ButtonParams extends BaseParams { title: string; label?: string; }
  export interface FolderParams extends BaseParams { title: string; expanded?: boolean; }
  export interface ListParamsOptions<T> { [label: string]: T; }
  export type ArrayStyleListOptions<T> = { text: string; value: T }[];
  export type ObjectStyleListOptions<T> = { [label: string]: T };
  export interface BooleanInputParams extends BaseParams { label?: string; }
  export interface NumberInputParams extends BaseParams { label?: string; min?: number; max?: number; step?: number; }
  export interface StringInputParams extends BaseParams { label?: string; options?: ObjectStyleListOptions<string>; }
  export interface BooleanMonitorParams extends BaseParams { label?: string; }
  export interface NumberMonitorParams extends BaseParams { label?: string; }
  export interface StringMonitorParams extends BaseParams { label?: string; }
  export interface ColorInputParams extends BaseParams { label?: string; }
  export interface Point2dInputParams extends BaseParams { label?: string; }
  export interface Point3dInputParams extends BaseParams { label?: string; }
  export interface Point4dInputParams extends BaseParams { label?: string; }
  export interface TabParams { pages: { title: string }[]; }
  export interface TabPageParams { title: string; }
  export type TpPlugin = unknown;
  export type TpPluginBundle = unknown;
  export type PluginPool = unknown;
  export class TpChangeEvent<T> { readonly value: T; readonly presetKey: string; }
  export class BladeApi<C = unknown> {}
  export class ButtonApi extends BladeApi {
    on(eventName: 'click', handler: (ev: unknown) => void): this;
  }
  export class InputBindingApi<In, Ex> extends BladeApi {
    on(eventName: string, handler: (ev: TpChangeEvent<Ex>) => void): this;
  }
  export class ListInputBindingApi<T> extends InputBindingApi<T, T> {}
  export class MonitorBindingApi<T> extends BladeApi {}
  export class SliderInputBindingApi extends InputBindingApi<number, number> {}
  export class TabApi extends BladeApi {}
  export class TabPageApi extends BladeApi {
    addBinding<O extends object, K extends keyof O>(object: O, key: K, params?: BindingParams): InputBindingApi<O[K], O[K]>;
    addButton(params: ButtonParams): ButtonApi;
    addFolder(params: FolderParams): FolderApi;
  }
  export class FolderApi extends BladeApi {
    addBinding<O extends object, K extends keyof O>(object: O, key: K, params?: BindingParams): InputBindingApi<O[K], O[K]>;
    addButton(params: ButtonParams): ButtonApi;
    addFolder(params: FolderParams): FolderApi;
    addBlade(params: BaseBladeParams): BladeApi;
    addTab(params: TabParams): TabApi;
  }
}
