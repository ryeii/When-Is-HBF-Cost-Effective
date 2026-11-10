# When Is High-Bandwidth Flash Cost-Effective for LLM Inference?

This repository contains the simulation artifact for the [preprint](https://arxiv.org/pdf/2611.11036). The study compares conventional memory and offload, shared-site HBF, and cascaded HBF across nine dense and MoE models, two GPU platforms, five workloads, and installations from 1 to 100 GPUs. It asks how HBF changes the installed cost at fixed demand and useful throughput at fixed GPU count.

## Quick start

Use Python 3.11 on Apple silicon macOS or on Linux with a compatible pinned wheel (x86-64 or AArch64, glibc 2.28 or later). The reference runs and release checks used Apple silicon. The experiment runs on the CPU and requires no model weights, GPU, or Hugging Face login.

```sh
./scripts/setup.sh
.venv/bin/python scripts/reproduce.py smoke
.venv/bin/python scripts/reproduce.py figures
```

The smoke run simulates one short-chat cell each for Qwen3-8B and Muse Spark Open. The figures command rebuilds all seven figures from the archived results and writes them to `runs/figures/`.

## Run the full suite

```sh
.venv/bin/python scripts/reproduce.py full --workers 2
```

From another terminal, check progress with:

```sh
.venv/bin/python scripts/reproduce.py status
```

The full run computes the 720 original cells and 90 Muse cells, the fixed-demand candidate expansion, boundary replays, sensitivity corners, held-out routing intervention, and Muse timing checks. It analyzes the fresh outputs and generates all seven paper figures. The full run is CPU- and disk-intensive. Allow several hours or longer and keep at least 15 GB free. Start with one or two workers on a memory-constrained machine.

Results, stage logs, and status are written under `runs/full/`. Use `--name NAME` on both commands for a separate run. Reissue the same command after an interruption to skip completed stages. The main sweep can carry forward verified completed model/platform jobs, and the Muse matrix checks completed cells individually. The other interrupted stages restart from the beginning. Keep the run directory in place since receipts reference earlier attempts by path.

The optional accelerator compiles the same event-model source and checks complete event trajectories against the Python path before use. It requires a C compiler and Python development headers. Set it up before starting the full run.

```sh
.venv/bin/python -m pip install Cython==3.2.1
.venv/bin/python scripts/reproduce.py compile
.venv/bin/python scripts/reproduce.py tests
```

## Repository guide

| Location | Contents |
| --- | --- |
| [`scripts/`](scripts/) | Environment setup, experiment commands, and figure generator |
| [`experiments/hbf_adoption/`](experiments/hbf_adoption/) | Simulation and analysis code, contracts, methodology, frozen timing profiles, model geometries, workload shapes, and routing inputs |
| [`evidence/`](evidence/) | Figure data, archived worker outcome rows, provenance and selected logs |
| [`third_party/`](third_party/) | Upstream data and software notices |

The archived numeric results and the local run commands are independent of the historical GitHub archive. The [evidence guide](evidence/README.md) maps the published figures and evidence to their source runs.
