# Experiments

## Ideas

-   Different models with different sizes (e.g. 1B/7B/70B)
-   Varying and heterogeneous hardware (high/low end GPUs, pure CPU)
-   Measure latency and throughput (single token vs full inference vs multiple parallel inference requests)
-   Vary partitioning techniques (e.g. how small are split models, better to keep larger chunks?)
-   Vary client behavior
    -   Different number of clients (e.g. 5/15/50 clients)
    -   Frequent disconnects
    -   Clients with bad hardware (i.e. slow)
    -   Clients with bad network
-   Resource utilization (% of GPU at load, memory, idle time, ...)
-   Network
    -   bandwidth
    -   packet loss
    -   latency
    -   failure rates
    -   e.g. run experiment with low bandwidth clients -> model chunks will take longer to load
-   Comparison: Ours vs. ONNX, WebLLM, WeInfer, ...
