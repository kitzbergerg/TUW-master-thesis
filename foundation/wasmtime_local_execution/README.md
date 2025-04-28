Run with:

```
cargo component build
wasmtime run -Snn --dir data/::data target/wasm32-wasip1/debug/foundation_local_model_execution.wasm
```

Cargo component:

```
cargo binstall cargo-component
```

Wasmtime needs to be installed with:

```
cargo install --features component-model,wasi-nn,wasmtime-wasi-nn/onnx --path .
```
