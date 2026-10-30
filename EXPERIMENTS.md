# Development experiments

## 0. Official starter, measured on Kaggle T4

Source: user-provided `baseline_results.zip`, one run of each side, seed 1.

| Metric | Base | Official SwiGLU example |
|---|---:|---:|
| Full command, seconds | 82.43 | 85.40 |
| Peak GPU memory, MiB | 14345 | 12385 |
| Training loop, seconds | 56.72 | 61.25 |
| Public eval loss | 2.04969 | 2.04965 |

Checks 1, 2, 5–10 passed; 3 and 4 failed as expected (no own kernels).
No profile was recorded. These numbers are NOT results of our RMSNorm implementation.

## 1. Frozen-weight RMSNorm — pending GPU validation

Files: `solution/rmsnorm.py`, `solution/test_rmsnorm.py`.

Hypothesis: replace the chain of RMSNorm reductions, casts and elementwise operations
with one forward and one backward kernel per norm, reducing intermediate storage
and launches. Keep the official SwiGLU example enabled; its time does not count
toward our required 5%.

Implementation preserves Qwen2's cast before weight multiplication, output dtype
promotion, and backward rounding at the cast. Only frozen weights are supported;
CPU, trainable weights and unsupported inputs fall back to Transformers.
Class patch is idempotent.

Reference expression (not a third-party kernel implementation):
https://github.com/huggingface/transformers/blob/v5.16.1/src/transformers/models/qwen2/modeling_qwen2.py

Local checks: Python AST parsing and `git diff --check`.
Not yet checked: Triton compilation, numerical tests on GPU, full training,
kernel share, speed, memory and adapter quality. No improvement is claimed yet.

### Kaggle run

Use the provided notebook and set:

```python
REPO_URL = "https://github.com/Kafka3141/test-repo.git"
COMMIT = "codex/soup-starter"
```

The branch is for iteration; final submission must pin a commit hash.
Use T4 x2, Internet enabled. Run the correctness tests before full measurements:
`python -m pytest solution -q`. If they fail, stop and save the complete error;
do not treat later harness output as a validated result.

After a full run, preserve result.json, checks.json, the run metrics and logs.
For diagnosis only, the harness supports `--profile` (see its README). Profiled
runs are separate from measurements and are not evidence of final performance.

The working repository was created by the participant. Before submission the
solution must be transferred to the repository issued by the hackathon platform.
