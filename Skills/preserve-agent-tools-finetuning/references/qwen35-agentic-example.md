# Qwen3.5 4B: base vs adapter-f16_final

Evaluation complete.

## Configuration

Context 262144; reasoning enabled with server budget -1; temperature 0.8; top_p 0.95; top_k 20; min_p 0; seed 42. One attempt per task and model. Both models use the same source snapshot, task prompts, sample and limits. The configured context is 256K; these short tasks do not test effective use of 256K input tokens.

## Repository tasks — executable outcomes

Local tool harness, not VS Code Copilot. A pass requires read/edit/test workflow, normal completion, reasoning, and independent public + hidden regression checks. No partial credit.

| Task | Repair | Base | Adapter final |
|---|---|---|---|
| S1 | Fix RemoveNanValues | FAIL: turn_limit | FAIL: repeated_tool_loop |
| S2 | Fix RemoveDuplicatedValues | FAIL: repeated_tool_loop | PASS |
| S3 | Fix RemoveEventBasedDuplicatedIndexes | FAIL: turn_limit | FAIL: turn_limit |
| S4 | Fix SignalIteratorBase | FAIL: turn_limit | FAIL: turn_limit |
| S5 | Fix GetArrayIndices on an empty path | FAIL: turn_limit | FAIL: repeated_tool_loop |
| M1 | Fix NearestTimestampIterator skip_till_first_rpu behavior when REF/RPU timestamp counts differ. If no RPU timestamp matches REF within the threshold, the iterator must be exhausted. Empty RPU must also be exhausted when skipping; matches at REF index zero and in the middle must work. | FAIL: repeated_tool_loop | PASS |
| M2 | Fix CalculateMessageStatistics so missing timestamp counts cannot wrap around when comparison count exceeds an input count. Clamp a negative difference to zero. Preserve the existing meaning | FAIL: turn_limit | FAIL: turn_limit |
| M3 | Fix FindNearestValueInWindow at uint64 timestamp boundaries. A window extending below zero or above UINT64_MAX must be clamped to the representable range, and timestamp zero with zero deviation must work. Preserve existing value-range/interpolation behavior and return semantics. | FAIL: turn_limit | FAIL: turn_limit |
| M4 | Fix NearestTimestampSignalIterator | FAIL: repeated_tool_loop | FAIL: turn_limit |
| M5 | Fix ArrayWrapper to recognize and group two-dimensional flattened fields such as matrix[0,0], matrix[0,1], matrix[1,0]. Return correct dimensions and item values with numeric row/column ordering even for indices >=10. Preserve one-dimensional arrays and empty/missing arrays. Arrays in scope have contiguous zero-based indices. | FAIL: repeated_tool_loop | FAIL: turn_limit |

| Repository total | Base | Adapter final |
|---|---:|---:|
| Passed / attempted | 0/10 | 2/10 |

## prop_bench — separate whole-file edit evaluation

50 fixed records from the supplied validation dataset, stratified sampling with seed 42. The benchmark’s contradictory default system prompt was replaced with one identical explicit whole-file output instruction for both variants. Initial pilot outputs are excluded.

| Metric | Base | Adapter final |
|---|---:|---:|
| Tasks | 50 | 50 |
| Normal, nonempty completion | 44 | 46 |
| Strict successes: direct reviewer estimate plus exact matches | 11 | 14 |

Paired normalized outcomes: both accepted 8, base only 5, adapter only 6, neither 31. Base had 4 repeated-tool loops and 6 turn limits on repository tasks; adapter had 2 loops, 6 turn limits, and 2 normal verified finishes. Base final code passed independent checks in 6 tasks, but the agent failed the completion gate in all 6. This is why code-only scoring would overstate agentic success.

The standard Copilot judge score is **unavailable**: the authenticated SDK could not connect through the proxy, and direct API checks also failed. Those errors are not model failures. The fallback review is performed by the Codex assistant, is not blind, and does not execute these external-project files. Per-record reasons are in manual_reviews.json. The executable repository score is the stronger evidence of task completion.

## Validation and artifacts

- All 10 regression checks fail against unchanged code and pass with reference fixes: grader_validation.json.
- Protocol controls verify that no-tool responses, repeated read loops, and edits after the last test cannot pass: harness_validation.json.
- Streaming-parser control verifies split tool arguments, multiple calls, reasoning and finish reasons: test_stream.py.
- Exact requests, raw responses, reasoning, tool transcripts, diffs and test outputs: agent_runs/ and prop_bench/.
- Model/server/dataset/source hashes and limits: manifest.json.
- Dataset reference defects and executable evidence: dataset_issues.json (scored against the instruction, not the defective reference).
- Snapshot file hashes: snapshot_manifest.json.
- Protocol details and limitations: PROTOCOL.md.

The adapter has the stronger agentic result under this protocol (2 vs 0), while the normalized prop_bench edit result differs by only one case and does not establish a reliable advantage. One sample per task at one seed does not establish run-to-run reliability. No combined score is computed across the two different benchmarks. Adapter training overlap with the supplied validation data was not independently verified.
