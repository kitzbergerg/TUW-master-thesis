# Democratizing LLM Inference Through Browser-Based Federated Computing

## Overview

Large Language Models (LLMs) are currently accessible primarily to organizations with substantial computational resources. This thesis explores democratizing access to large models by distributing inference across web browsers using WebAssembly (WASM), enabling anyone to contribute resources and access models that would otherwise be computationally infeasible.

## Core Idea

Instead of requiring expensive GPU clusters, LLM inference can be split across multiple browser instances using WebAssembly and WebGPU. While large open models like LLaMA 70B, DeepSeek R1, or Kimi K2 are publicly available, they remain inaccessible to individuals due to their computational requirements (often exceeding 100GB of VRAM). By leveraging distributed browser-based computing, users can contribute their device's resources to collectively serve these models, enabling anyone to test and interact with state-of-the-art LLMs without requiring substantial hardware investments.

## Key Research Areas

### Distributed Systems & Orchestration

**Tasks**

-   Browser-based client coordination and resource discovery
-   Dynamic workload distribution and load balancing strategies
-   Fault tolerance and graceful handling of client disconnections
-   Scalable architecture for heterogeneous device capabilities
-   Real-time adaptation to changing client pool composition

**Research questions**

-   How can workload be optimally distributed across heterogeneous web-based devices?
-   What client information is necessary for effective workload management decisions?
-   How can the system adapt to dynamic changes in available computational resources?

### LLM-Specific Optimizations

**Tasks**

-   Transformer model partitioning across computational graph boundaries
-   Distributed KV-cache management for autoregressive generation
-   WebGPU acceleration and memory-efficient tensor operations
-   Model quantization strategies for diverse browser environments
-   Optimized communication protocols for large tensor transfers

**Research questions**

-   What are the optimal model partitioning strategies for different network topologies and device configurations?
-   What is the best approach for managing a distributed KV-cache?
-   How can model execution be optimized for the browser (WebGPU, buffer management, quantization, ...)?

### Cross-Domain Research Questions

-   How does browser-based federated inference compare to traditional centralized approaches in terms of latency, throughput, and resource utilization?
-   How can communication be optimized for varying network conditions (binary protocols, compression, transfer of large tensors, websockets)?

## Technical Implementation

-   **Frontend**: WebAssembly with WebGPU/CPU execution using ONNX Runtime Web
-   **Backend**: Webserver managing workload based on connected clients
-   **Communication**: WebSocket-based coordination with efficient binary protocols
-   **Model Splitting**: Partitioning transformer models across computational graph boundaries

## Experiments

See [experiments](experiments.md) for details.
