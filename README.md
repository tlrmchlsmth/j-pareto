# j-pareto

Benchmarking automation for exploring when prefill/decode disaggregation makes sense in [llm-d](https://github.com/llm-d/llm-d) deployments.

## What This Does

LLM inference can run as a single aggregated server or split into dedicated prefill and decode stages (P/D disaggregation). P/D improves interactivity (lower TTFT) at the cost of GPU efficiency. The key question is: **at what concurrency levels does P/D actually win?**

This repo automates that comparison by:

1. **Deploying** aggregated and disaggregated configurations side-by-side on Kubernetes
2. **Sweeping** across concurrency levels with calibrated benchmarks (auto-sizing run duration for statistical stability)
3. **Plotting** Pareto frontiers of interactivity (throughput-per-user) vs. efficiency (throughput-per-GPU) to show where each architecture excels

## Quick Start

```bash
# Set up environment
cp .env.example .env  # Add HF_TOKEN and GH_TOKEN

# Run a Pareto sweep comparing agg and disagg configs
cd pareto
just run-all           # Deploys, benchmarks, tears down each config
just collect-sheets "My Sweep Results"  # Upload results to Google Sheets
```

## Repository Structure

```
agg/          Aggregated deployment template (single model server, no P/D split)
disagg/       Disaggregated deployment template (prefill + decode with NIXL KV transfer)
pareto/       Pareto frontier sweep framework
  configs/    Sweep configurations (TP, DP, replica counts)
  collect.py  Collect results from K8s Job logs -> CSV / Google Sheets
  job.yaml    K8s Job template for running calibrated sweeps
pd-config/    Systematic TP/DP parallelism exploration framework
poker/        Interactive benchmarking pod (Dockerfile + K8s manifest)
```

## How the Pareto Sweep Works

Each config in `pareto/configs/` defines a deployment variant (e.g. `agg-tp4` = 4-GPU aggregated, `disagg-1p1d-8` = 8-GPU with 1 prefill + 1 decode). The sweep:

1. Deploys the config via Helmfile using the `agg/` or `disagg/` template
2. For each concurrency level (4, 8, 16, ... 128):
   - Runs a short **calibration** to estimate throughput
   - Computes how many prompts are needed for a target duration (default 60s)
   - Runs the **main benchmark** with the computed request count
3. Collects structured logs and computes:
   - **TPSU** (throughput per user) = output tok/s / concurrency
   - **TPSG** (throughput per GPU) = output tok/s / GPU count
4. Plots TPSU vs TPSG as a Pareto frontier

The result shows the interactivity-efficiency tradeoff: P/D configs typically win at low concurrency (better TPSU) while aggregated configs win at high concurrency (better TPSG).

## Requirements

- Kubernetes cluster with GPU nodes (AMD MI300X or NVIDIA)
- `kubectl`, `helm`, `helmfile`, `just` CLI tools
- `.env` file with `HF_TOKEN` and `GH_TOKEN`
