# Simulation-only execution protocol

The study uses public measurements and literature to anchor a CPU simulation. The manuscript reports conditional deployment projections and distinguishes them from measured inputs and narrower reference runs.

## Evidence levels

1. **Measured inputs:** released GPU/operator measurements, official architecture/configuration snapshots, public routing and workload data. Preserve source revisions, units, software/backend and geometry.
2. **Executed reference anchors:** five fresh LLMServingSim/ASTRA reference replays; actual non-dummy Frontier H200/Qwen3-235B TP8 run; matched resident/native-HBF/host/NVMe serial expert pilots; detailed HBFSim device runs. These establish execution and consistency in their stated domain, with the already reported fidelity limitations.
3. **Conditional deployment projections:** a fast event simulation spanning the wider model, architecture, demand and 1–100 GPU space. Public-data fits supply timing primitives; extrapolations are explicit and varied. Report outcomes under the stated simulator assumptions, not validated product predictions.
4. **Sensitivity stability:** a conclusion surviving all tested discrepancy/physical/cost scenarios is stable within that tested set. This is neither a confidence interval nor proof against unmodeled effects.

The original strict evidence gates still define what may be called validated. They must not make every unmeasured counterfactual unreportable. Use a distinct projection classification and never turn missing execution into zero throughput, infeasibility or a real-world enabling claim.

## Composite model and literature reuse

- Detailed serving anchor: Frontier's public operator tables, MoE per-layer EP waves, host preparation/launch semantics and actual event clock. The narrow serial cold-expert adapter leaves upstream unchanged.
- Detailed memory anchor: native HBFSim transaction engine and the measured-data-calibrated A100 external paths. Build a fast service surrogate for large sweeps, and separately report its agreement with native cases. Finite request granularity, staging buffers and processor links remain explicit.
- Fast GPU primitives: rerun the existing actual LLMCompass roofline method / released LLMServingSim composition and public NVIDIA AISimulate GEMM fits. Fit attention separately from released RTX4090 and H200 measurements with entire-shape holdouts. Reuse Frontier CPU preparation/output costs and eager host launch floors; exclude forward-drain fields that can contain GPU waits. No legacy adoption results are imported.
- Model payloads: `model_catalog.py` derives eight text architectures from immutable official configurations or the pinned public Meta architecture source. It explicitly counts MLA projection geometry, GQA/MLA KV storage, TP replication and PP layer partitioning. The common BF16 representation for FP8-released DeepSeek/Kimi architectures is a counterfactual format, not the downloaded checkpoint's serialized size or a quality claim.
- Routing: use released Qwen3-30B Q4 route arrays as a separately labeled fixed-route intervention. Other checkpoints need declared routing scenarios, including uniform and skewed controls; Qwen observations are not measured Kimi/DeepSeek routes. Only training-partition statistics may choose static hot experts.
- Prefixes: exact ancestor-sensitive block identity from public teacher-forced text for identity-sensitive experiments. TraceLab contributes observed length/gap shapes and recoverable causal links, not invented token identities. Model/format/template changes invalidate keys.
- Conventional controls: resident, host-DRAM cache and multi-drive SSD staging; sweep host capacity and drive provision. Include an explicitly optimistic CPU-expert execution bound motivated by COPE rather than treating serial GPU staging as the strongest alternative. An oracle bound is not a realizable measured baseline.

## Deployment implementation requirements

Use actual TP/EP/PP/replica layouts. A 100-GPU installation occupies 13 eight-slot nodes with four empty slots on the final node; it does not imply TP100. Layer stages, replica routing, per-node DRAM/SSD/ingress and network resources must share clocks. Avoid projecting one-GPU throughput by multiplying by GPU count.

The fast event model must reveal transfers when their producer/routing dependencies complete. Reserving a future stage's I/O before it is ready can introduce anticipation and artificial contention. Keep bounded staging allocations, completed-prefix state, active KV growth and failed/censored requests explicit. Serial staging is a realizable control; overlap requires declared buffers and shared-bandwidth accounting.

Shared-site projections scale near capacity and bandwidth by the retained-site fraction and model the changed kernel memory service. The H3-like projection retains near memory, charges staging writes and normal GPU consumption separately, and caps HBF service by an independent processor link. Count installed resources in both. Neither architecture is a retail GPU retrofit or a faithful H3 reproduction.

## Workloads and decisions

Cover short chat, long/shared-prefix prompts, tool/agent shape scenarios, paused sessions and offline work under responsive/balanced/relaxed interactive targets. Keep closed-loop successor release separate from externally offered load. Report fixed-demand, proportional-demand and useful-capacity views; reclassify common request clocks for different SLOs where possible rather than rerunning identical timings.

Main results should include absolute TTFT/mean ITL/p99 ITL, joint attainment, successful requests/s and output goodput, resource/capacity/traffic decomposition, the strongest conventional comparator and unsupported/extrapolated scope. Use at least multiple deterministic workload seeds/windows and expose sensitivity near boundaries.

Unknown HBF prices are handled by affordable **total installed premium**, in declared GPU-equivalent or acquisition-scenario units. Host DRAM/SSD costs and CPU alternatives must be varied or bounded; free conventional offload resources are a useful adversarial economic control. Report no market-dollar or TCO headline without suitable inputs. Re-optimize layouts at every price and physical scenario.

## Corrections retained

The first identity workload used a hardcoded 32-layer KV byte conversion for Qwen3-8B, which has 36 layers. Its token identities remain inspectable, but all dependent physical-byte/capacity reports are marked superseded. The corrected generator derives KV bytes directly from the pinned configuration; the corrected cache sweep holds physical GiB capacities fixed and charges fractional-block tails.

## Executed deployment reference

The final fast GPU reference uses compiled AISimulate native operation graphs and separately pinned H200 operator tables, not the standalone attention fit for MLA. The binary is version 0.12.0.dev2 and has its own wheel hash; the newer checked-out source is not falsely claimed to be the executed build. Public full-forward measurement manifests were checked and do not match these eight BF16 cases. Consumer component transfer remains an explicit extrapolation.

The main conventional control grants 256 GiB per consumer host domain or 4 TiB per H200 domain plus eight SSDs, all free in the acquisition comparison, and includes the zero-arithmetic CPU bound. The tested physical sensitivity includes faster conventional I/O; there is no invented dollar HBF price. Every price point reselects feasible designs and charges all installed stacks.

Each finished response excludes its last sampled token from produced KV until a subsequent forward consumes it. Far copies persist on promotion; new admissions pay write and drain service. Exact held-out routing is a separate intervention using current batches and training-only static identities. Independent node copies are deterministic program copies, never independent statistical samples. Heterogeneous tool useful-capacity comparisons are restricted to replica widths at most eight; fixed-four-user programs remain matched across widths.

Optimistic physical/source/stream-slot bounds and a fractional root-cohort resource-packing LP resolve overloaded targets without assigning throughput. Every complete event trajectory checks the bounds against its clocks. Event-guard failures are retained and retried, never recoded as infeasible. Reused complete trajectories require structural equality of all event physics and unchanged data; the bound-only method may change, and original source lineage is retained. See the final manifests for exact coverage and exclusions.
