# Federated Inference Using WebAssembly (WASM)

## Overview

Federated inference allows multiple distributed devices to collaboratively perform machine learning inference without requiring centralized computation.
This approach leverages WebAssembly (WASM) to enable users to contribute computational resources through their browsers.
Users wishing to utilize a model must first contribute by computing tasks for other users, leveraging WASM alongside WebGPU or CPU execution.

## Key Idea

Instead of running inference on a single device (which may be slow or infeasible due to model size), inference tasks can be distributed across connected clients, improving efficiency and scalability.
A potential application is serving a large language model (e.g., LLaMA) across multiple instances, distributing workloads dynamically.

## Advantages

-   Accessibility: WASM runs on various platforms, including browsers and edge devices.
-   Ease of Deployment: Users only need to visit a website to participate.
-   Privacy-Preserving: Computation can be split across nodes to limit data exposure.

## Challenges

-   Scientific Contribution: Requires validation against existing distributed inference methods.
-   Task Distribution: Managing load balancing, failures, and varying computational capabilities.
-   Model Partitioning: Ensuring efficient model distribution and inference performance.

## Research Questions

-   How can computation be effectively distributed across browser instances?
-   What strategies optimize the allocation of computational tasks based on client device capabilities and network conditions?
-   What incentive mechanisms ensure fair and sustainable resource contribution from participants?

-   How can large language models be effectively partitioned for distributed inference across browser instances?
-   How does federated inference using WASM compare to running models natively? (E.g. compare federated with swapping model parts in GPU)
-   What are the limitations and advantages of browser-based inference compared to containerized environments?

## Split

Software engineering:

-   task splitting
    -   How to handle low number of connected users?
    -   How to handle dropouts?
    -   How to distribute workloads and rebalance?
    -   How to find best assignment of tasks to nodes? How to handle slower/faster participants? How to handle network latency? Something like 'task x node' matrix to find best assignments?
-   computational graph (i.e in which order to compute what and where to send the result)
    -   How can computation be distributed and gathered?
    -   DAG?
-   plugin system
    -   Should we build a plugin system to allow users to distribute arbitrary workloads?
    -   What are the interfaces? WASM modules?
    -   How can a webserver serve arbitrary wasm modules and data for computation? Websockets? What about larger data packets like AI models?
-   computational rewards
    -   How can we ensure that only users that contribute can use resources?
    -   How to ensure fair distribution?
    -   How to calculate rewards (some users have different hardware, e.g. GPU vs CPU for AI)?

Data science:

-   implement plugin for federated inference
-   How to keep model across sessions? Users shouldn't have to download large amounts of data multiple times. Use File System Access API or IndexedDB.
-   How can we optimally split an ONNX model for federated inference? ONNX is DAG already, can we just reuse it?
-   How does a cold start compare to a running instance? I.e. for a cold start all users would have to download model weights.
-   What is a good model size (MBs vs GBs)?
    -   Probably depends on focus: Long running instances prefer large models (don't care about long download), short instances prefer small models (faster download)
    -   Does it make sense to serve large models (>4GB)? How can we work with WASMs 4GB limit in this case? Move to GPU in chunks?
    -   Use quantization?
-   Performance:
    -   Caching:
        -   Important for LLMs since autoregressive.
        -   Is it possible to implement a variant with attention cache? This would also significantly reduce amount of data to be transferred. Might be complicated, there prob is nothing like that for WASM yet.
            -   Llama inputs/outputs kv. Onnxruntime-web supports keeping session outputs in GPU memory. Therefore by using onnxruntime this might not be as complicated.
        -   What about keeping a cache on the server? That way workers only have to send newly computed weights? When a worker drops out the whole server cache is sent to a new worker, otherwise a worker with intact cache is used only forwarding new data.
    -   Which libraries to use? wasm-nn, onnx-web, tensorflow.js, ...
    -   How low-level do we need to go? Is it ok to just use libraries or implement ML-framework from scratch? What about compiling frameworks like BLAS to WASM? How to optimally utilize GPU?

## Test data

-   AI models
-   EDDS paper: https://github.com/dilina-r/mcts-rec
-   Any other highly parallelizable task? (e.g. use framework instead of openmp)

## Roadmap

### Foundation

-   Exploration of technologies: Investigate wasm-nn, WebGPU, ONNX format and other technologies
-   Local model execution: Implement and benchmark a standalone ML model running in WebAssembly
-   Split small model manually. Run in different WASM modules manually. Verify that output is the same as non-split model. Goal: Understand model architectures better
-   Literature review: Review of distributed inference, WebAssembly performance, and LLM deployment techniques

### Basic distributed system

-   Web server implementation: Create a system to serve and coordinate WebAssembly modules
-   Basic task distribution: Implement simple task allocation across multiple clients
-   Performance measurement: Develop metrics and tools for measuring system performance

### Advanced features

-   Computational Graph Optimization: Develop an execution flow for distributed inference
-   Model splitting: Implement techniques for partitioning large models across multiple clients
-   Caching strategies: Develop and evaluate different approaches to caching for improved performance
-   Multi User: Allow for multiple users to send inference requests at the same time
-   Incentive mechanism: Design and implement a reward system for resource contribution

### Evaluation

-   Case study with LLMs: Implement and evaluate the system using models like Llama 3 or DeepSeek
-   Comparative analysis: Compare your approach with centralized inference and container-based distributed solutions
-   Benchmark development: Create reproducible benchmarks for federated inference systems

## Notes

-   Maybe interesting for edge devices? Since wasm essentially runs anywhere
-   Maybe test different platform (i.e. PC with GPU, Laptop, Mobile, ...). See how different combinations behave (PC/Laptop/Mobile only, mix, ...).
