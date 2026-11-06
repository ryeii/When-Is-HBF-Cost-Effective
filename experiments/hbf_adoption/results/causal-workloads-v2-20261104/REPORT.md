# Public agent-session workload audit

Audited 357,161 released rows, removed 514 identical normalized duplicate rounds, and converted 356,647 unique rounds in 4,265 sessions; retained 432,510 tool records in the source audit.

| Transition | Rounds |
| --- | ---: |
| root | 4,265 |
| parallel_tool_barrier | 266,942 |
| human_gap_proxy | 37,173 |
| unresolved | 48,267 |

Model switches: 132; input-length decreases: 7,867; reported prefix/new token accounting mismatches: 0.

Tool-barrier transitions use the maximum latency of the required predecessor tools; parallel tools are not summed. All tools release after simulated response completion in this declared conservative orchestration scenario. Human delays remain explicitly uncertain timestamp proxies. Missing dependencies are retained as unresolved.

The public release supplies lengths and cache-usage counters, not prompt text or token identities. No row is eligible for identity-based prefix admission. Applying these lengths to an open model is a shape scenario and does not reproduce the proprietary model’s tokenization, cache behavior, or generation quality.
