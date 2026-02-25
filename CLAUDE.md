# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Preferences

- Use `podman` instead of `docker` for container builds

## Repository Overview

Deployment configurations and benchmarking automation for [llm-d](https://github.com/llm-d/llm-d), a Kubernetes-native distributed inference serving stack for large language models. This repo provides two main benchmarking frameworks for comparing aggregated vs. prefill/decode disaggregated architectures:

- `agg/` - Aggregated deployment template (single monolithic model server)
- `disagg/` - Disaggregated deployment template (prefill/decode split with NIXL KV cache transfer)
- `pareto/` - Pareto frontier sweep framework: deploy configs, run calibrated benchmarks at varying concurrency levels, collect results
- `pd-config/` - Systematic TP/DP benchmarking framework for exploring parallelism configurations
- `poker/` - Interactive testing pod with pre-installed benchmarking tools
- `llm-d/` - Git submodule pointing to the main llm-d project
- `Justfile.remote` - In-cluster benchmarking commands (copied to poker pod)
- `collect.py` - Result collection script for log-based sweep results

## Common Commands

This repository uses `just` as the task runner.

### Environment Setup

The Justfile requires a `.env` file at the repo root with:
- `HF_TOKEN` - HuggingFace token for model access
- `GH_TOKEN` - GitHub token

### Top-Level Commands

```bash
just                    # List available commands
just create-secrets     # Create HF and GH token secrets in namespace
just start-poker        # Deploy the poker pod
just print-gpus         # Print GPU allocation across all nodes
just cks-nodes          # Print CoreWeave node details
just get-decode-pods    # Fetch and cache decode pod info
just copy-traces        # Copy PyTorch traces from decode pods
```

### Pareto Sweep Workflow

The primary workflow for comparing configurations:

```bash
cd pareto

# Run a single config end-to-end (deploy → wait → sweep → teardown)
just run agg-tp4

# Run all configs with GPU budget scheduling
just run-all

# Or step-by-step:
just up agg-tp4              # Deploy
just wait-ready agg-tp4      # Wait for model servers
just sweep agg-tp4           # Launch sweep job
just wait-sweep agg-tp4      # Wait for completion
just teardown agg-tp4        # Clean up

# Collect results
just collect                  # CSV output
just collect-sheets "Title"   # Upload to Google Sheets with Pareto chart
```

### Agg / Disagg Direct Deployment

For deploying individual configurations directly:

```bash
cd agg   # or cd disagg

just deploy              # Deploy the stack
just status              # Show pod status
just gateway-ip          # Get gateway IP
just start-poker         # Deploy poker pod
just poke                # Interactive shell in poker pod
just destroy             # Tear down
```

### In-Cluster Benchmarking

From inside the poker pod (via `just poke`):

```bash
just benchmark <target> <concurrency> <num_requests> <isl> <osl>
just benchmark_g <target> <concurrency> <num_requests> <isl> <osl>
just sweep <target> <isl> <osl> <concurrency_levels...>
just health <target>
just health-all
just profile <url>
just profile_all_decode
```

## Key Configuration Files

- `Justfile` - Top-level automation (GPU inspection, secrets, poker, traces)
- `Justfile.remote` - In-cluster benchmarking commands
- `agg/helmfile.yaml.gotmpl` - Aggregated deployment Helm orchestration
- `disagg/helmfile.yaml.gotmpl` - Disaggregated deployment Helm orchestration
- `pareto/Justfile` - Pareto sweep orchestration (GPU budget scheduler, config management)
- `pareto/configs/*.yaml` - Sweep configurations (TP/DP/replica combinations)
- `pareto/job.yaml` - Kubernetes Job template for running sweeps
- `pd-config/Justfile` - Systematic TP/DP benchmarking framework

## Important Notes

- Default model is `openai/gpt-oss-120b` (configurable via `MODEL` env var)
- Pareto sweep configs are in `pareto/configs/` — prefix determines base template: `agg-*` uses `agg/`, `disagg-*` uses `disagg/`
- `MAX_GPUS` controls concurrent GPU budget during `run-all` (default: 32)
- The poker pod image (`quay.io/tms/poker:0.0.13`) includes vLLM bench, guidellm, lm_eval, and kubectl
- vLLM API servers can take several minutes to start for large models
- PyTorch profiling traces are stored in decode pods at `/traces` and copied locally to `./traces/`
- Decode pod information is cached in `.tmp/decode_pods.txt`
