# Third-party inputs

- **AISimulate and NVIDIA operator timing data.** The CPU engine is installed from a pinned public wheel. Frozen native profiles and the vLLM 0.30.0 table overlay retain source and copy hashes in their manifests. See [AISimulate](https://github.com/ai-dynamo/aisimulate) and the accompanying Apache-2.0 license.
- **Cacheable-by-Design.** The Qwen3-30B Q4 routing array comes from commit `03366ef1cb986a7ac6067c840f8ac58d8e5c8595`. The MIT notice accompanies the released array. See [Cacheable-by-Design](https://github.com/Shriniwas410/cacheable-by-design).
- **TraceLab.** The frozen agent workload contains derived causal session shapes. TraceLab is credited under the accompanying CC BY 4.0 dataset terms. The shape audit, transform description, and source fingerprints are included with the workload. See [TraceLab](https://github.com/uw-syfi/TraceLab).
- **Model architectures.** The metadata manifest and the two architecture provenance files identify the official repositories and revisions for the geometry used in the simulator. The Llama and Muse snapshots contain extracted configuration facts. No checkpoint weights are distributed.
- **Other calibrations.** The device-service and projection-profile manifests, parameter evidence, and paper bibliography identify the HBFSim, Frontier and released measurement inputs used in the frozen service models.
