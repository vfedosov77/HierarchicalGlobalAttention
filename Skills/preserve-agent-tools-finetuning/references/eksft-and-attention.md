# Entropy/KL masking and proposed future-attention exception

## Paper versus local implementation

[Liu et al., EKSFT, arXiv:2605.29303](https://arxiv.org/abs/2605.29303) ranks target positions by model entropy and forward KL from a reference, masks high-ranked tokens from cross-entropy, and adds entropy/KL regularization on masked positions. Its reported experiments concern mathematical reasoning and later RL.

`Experiments/hga_qlora/src/train/run_train.py` implements **mask selection and masked CE**, not the paper's regularization losses. `eksft_lambda_entropy` and `eksft_lambda_kl` must be zero. The reference is the same frozen model with the LoRA disabled; ref and current-policy forwards run sequentially because the routed KV state is mutable. For each causal target `t`, statistics come from logits at `t-1`, including the carried previous-segment state. `k=int(N*rho)` positions with the largest entropy and KL are selected separately, then unioned by default; `labels=-100` excludes the union from CE. When KL is identically zero initially, only entropy selects tokens. Validation CE is unmasked.

The runnable A/B configs are `configs/train_qlora_rdiff_ntp.yaml` (SFT), `configs/train_qlora_rdiff_ntp_eksft.yaml` (masking), and `configs/train_qlora_rdiff_ntp_maskcontrol.yaml` (random-mask control); `scripts/run_eksft_ab.sh` runs the three arms, and `scripts/eval_eksft_drift.py` measures held-out CE, entropy, and KL. Read `Experiments/hga_qlora/docs/eksft-sft.md` before changing them. The local 20-record smoke measured about 28.8% effective masking at `rho=0.2` with a union, which is why the random control uses the **measured** mask share. These smoke numbers do not establish task or agentic gains.

## Proposed attention-aware exception: not merged

Motivation: a rare project fact can have high entropy yet be needed repeatedly by later tokens. Pure entropy masking can remove precisely the CE signal for that fact. The proposed rule is:

```text
H = top-rho entropy target positions
K = top-rho KL target positions
A = top-20%-future-attention target positions
mask = K union (H minus A)
```

Thus a token in `A` is exempt from **entropy-based** masking; high KL can still mask it. Define the future-attention score for a source token `i` as the sum of attention probability from supervised future query positions `j>i` to key `i`, over a stated set of full-attention layers/heads. Normalize for available future-query count or compare only positions with similar exposure; otherwise early tokens win merely because they have more future queries. Choose whether immediate neighbors are excluded and document it. Rank `A` among eligible supervised targets **per record**, with a fixed tie rule. For hybrid Qwen, GDN layers do not expose standard token-to-token attention; score the full-attention layers only.

An implementation must collect this score from the **unmasked** reference/current forward at the same checkpoint used for H/KL, across all segments without dense `S x S` materialization. HGA routing observes only selected historical chunks: attention to unselected chunks is not measured, so the score is a routed-attention proxy, not exact global attention. Accumulate per-key attention mass online, include within-chunk and cross-segment queries, and map key positions back to target IDs. Do not count a token's own query, future leakage, padding, or unsupervised prompt/tool tokens as candidate targets. Decide explicitly whether future queries can include unsupervised context; use one convention for all arms. Avoid leaking labels from validation into training-time selection.

Unit-test toy attention matrices where an uncertain high-attention fact is retained unless also high KL; test low-attention high-entropy masking, KL override, ties, segment boundaries, causal alignment, partial chunks, and bounded memory. Then compare four matched arms: ordinary SFT, existing EKSFT, attention-aware EKSFT, and a random mask matched to the **new effective** mask rate. Measure fact-token retention, unmasked validation CE, project QA/edit outcomes, and agentic tool tasks. Do not claim benefit from this rule until these measurements exist.
