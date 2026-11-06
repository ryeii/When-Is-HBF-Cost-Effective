# Published run evidence

The paper source and embedded figures come from Overleaf revision `c7b5089760e81f1e062d49ffa07682bf24c0f74d`. The original experiment record is commit `b2aab889d95977cb830200e2f50c7ae424d5ddf3` in the [historical archive](https://github.com/ryeii/HBF-Research). The archive requires repository access. Figure regeneration and the full simulation use only this checkout and the public dependencies installed by the setup script.

| Data | Used for |
| --- | --- |
| `adoption/minimum-frontiers.csv`, `adoption/price-curves.csv`, `muse.json.gz` | Minimum GPU count maps, workload margins, and premium curves |
| `adoption/aggregate.json.gz`, `muse.json.gz` | Example decisions and single-user feasibility |
| `adoption/scale-frontiers.csv`, `muse.json.gz` | Fixed-count throughput comparisons |
| `uncertainty/sensitivity-frontiers.csv` | Sensitivity audit figure |
| `uncertainty/route-intervention.csv`, `checks/` | Routing and timing checks |
| `worker-outcomes/`, `logs/` | Worker-level outcomes and selected historical execution logs and receipts |

The archival worker bundles are gzipped tar files. For example, to inspect the main matrix:

```sh
mkdir -p runs/archive/central
tar -xzf evidence/worker-outcomes/central.tar.gz -C runs/archive/central
```

The archival receipts keep the paths and hashes of the original execution. They can reference materials in the historical archive such as earlier attempts, original screening ledgers, and per-token trajectory files. Those bulky files are outside this checkout. Run the full suite to generate fresh screening ledgers, trajectories, and receipts locally.

## Historical run paths

All paths below are relative to `experiments/hbf_adoption/results/` at the archive commit.

- Central, boundary, and sensitivity: `terminal-study-20261105T183101Z/{central,corrections,sensitivity}`
- Fixed-demand expansion: `terminal-study-20261105T052432Z/expansion`
- Original-model figure tables: `terminal-study-20261105T183101Z/postprocess-20261105T192152Z/{adoption,uncertainty}`
- Muse: `muse-operator-matrix-20261105/` and `muse-operator-results-20261105.json`
- Route intervention: `route-intervention-20261104/`
