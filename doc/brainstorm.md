# Brainstorm

## Directions

-   Optimize current approach:
    -   GPU buffer reuse
    -   Optimize latency/throughput/network/...
    -   Store models after disconnect. Reuse on reconnect
    -   Duplicate/Redundancy of model blocks in case enough clients are connected (faster migration on disconnect)
    -   Clean disconnect: Migrate kv cache instead of rebuilding
    -   Pipeline parallelism: What about multiple requests? Web workers? Sync kv cache to server in background (is this even more efficient than just recomputing https://arxiv.org/abs/2403.01876)?
    -   Recompute only disconnected kv, extract other caches instead of recomputing everything
    -   Quantization
-   More complex computations:
    -   Better/automated model splitting
    -   Pipeline parallelism
    -   More complex ML architectures e.g. MoE
    -   Arbitrary ML (operate on ML computational graph directly instead of building blocks)
    -   Arbitrary computation (anything that runs in WASM; would require packing ML runtime into custom WASM module)
-   Batch processing
-   Allow different models in parallel
-   Use server as inference node as well

## Tradeoffs

-   Disconnecting clients vs Stable long running clients (relevant for cold-start, block redundancy, repartitioning, ...)
-   Latency vs Throughput (relevant for batching, cold-start, repartitioning, ...)
-   Single vs Parallel inference requests (relevant for kv cache and buffer management, batching, ...)
-   Sequential inference vs Pipeline parallelism (i.e. iter-operator vs intra-operator parallelism; relevant for idle resources, throughput, ...; consider network speed, ...)

## Other

WebGPU inserts bounds check. They make computation slower. Disable by running chrome with `--enable-dawn-features=disable_robustness`. See [details](https://github.com/mlc-ai/web-stable-diffusion/?tab=readme-ov-file#comparison-with-native-gpu-runtime-limitations-and-opportunities).
