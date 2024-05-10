# NeuralEdge

## A Deep Learning framework for multi-modal edge computing

NeuralEdge is an early-stage DL framework (with a defined 3Y roadmap 2024-2026), written in Rust focused on **RESEARCH** related to multi-modal AI agents, with local data processing. It consists of its own Tensor, autodiff and Transformers implementations.

It is currently not ready for production. The codebase is currently in the process of refactoring for parallel computation (CPU & GPU).

It was tested only on extremely small datasets and language models. It needs time to mature.

Its long-term goal is to support:
- WebAssembly + WebGPU applications
- multiple computational platforms (Nvidia, AMD, Intel, Metal)
- continual (lifelong) multi-modal learning

## Simplified roadmap for 2024-2025

- [X] NeuralEdge CORE: Tensor, FeedForward layers
- [X] Neuroverse LANG: Basic support for Transformers implementation
- [ ] Paralllel computation: GPU Backend (WebGPU) + Rayon (CPU) - currently in progress
- [ ] NeuralEdge MoDE: Mixture of Depth Experts
- [ ] NeuralEdge AGENT: Agentic modules

## Usage

See neuroverse-example (small language model training with English proverbs)