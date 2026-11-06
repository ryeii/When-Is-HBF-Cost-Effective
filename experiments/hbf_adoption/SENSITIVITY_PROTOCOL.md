# Conditional-boundary sensitivity protocol

Frozen before reading complete adoption frontiers. These scenarios test a finite set of discrepancies and hardware counterfactuals. They are not a confidence region. Central runs use two workload windows (17/29); sensitivity uses window 17 and retains every tested outcome.

The anchor set spans both GPU scales, dense and MoE weights, a large model, long shared prompts and paused prefixes:

| Platform | Model | Workload |
| --- | --- | --- |
| RTX4090 scale | Qwen3-32B | short_chat |
| RTX4090 scale | Qwen3-235B-A22B | short_chat |
| H200 | Llama-3.1-405B | short_chat |
| H200 | Kimi-K2 | short_chat |
| RTX4090 scale | Qwen3-8B | long_rag |
| H200 | Qwen3-8B | paused_personal |

At all nine installed counts, re-evaluate the central screened candidate pool for four fixed users. Re-select the conventional and HBF designs for every SLO and physical scenario. This is re-optimization within the evaluated pool, not a new exhaustive search. Preserve service-bound certificates separately from actual token-clock replays. Do not count certificates as observed throughput.

Scenarios, relative to the central assumptions:

1. HBF processor link limited to 64 GB/s.
2. HBF processor link limited to 256 GB/s.
3. HBF reads 25% slower, native GPU components 50% slower, cold compute fragmentation factor 2.
4. Conventional I/O service twice as fast and GPU components 25% faster. This is a deliberately favorable comparator counterfactual, not a claimed current-host measurement.
5. Skewed MoE inclusion instead of uniform; current/prior placement only. No route data is invented for the large model.
6. HBF program service 2 GB/s/stack and write amplification 4.
7. HBF program service at the native energy model's sustainable ceiling (at most 15 GB/s/stack with the main power budget), write amplification 1.
8. HBF self-heating budget 10 W/stack rather than 20 W/stack.
9. Combined adverse case: HBF link 256 GB/s, slower reads/GPU/fragments as in scenario 3, and faster conventional control as in scenario 4.

Program-service overrides remain capped by the stated media energy/power budget and are recorded explicitly. GPU timing multipliers do not erase near-memory physical service floors. Baseline and HBF software policies remain otherwise identical to the central run.

A separate exact-route intervention uses the released general-domain Qwen3-30B routes, with training-only static hot experts, held-out cyclic token positions and current-batch expert unions. Its BF16 payload is a controlled routing/format transfer from released Q4 traces, not a claim about measured BF16 routing. Compare its results with the marginal-inclusion model. Do not use those routes to validate Kimi or DeepSeek.

Report positive and negative affordable-premium results, no-feasible-candidate cases and which statements survive all tested scenarios. Unknown NAND endurance/retention is handled in a distinct arithmetic envelope: P/E budgets 1,000/3,000/10,000; refresh intervals 6 h/24 h/7 d; write amplification 1/1.3/4. Those envelopes are not vendor lifetime predictions.
