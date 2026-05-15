# Zero padded model inputs before execution

## Purpose

Fix stale padded model inputs in V1 `GPUModelRunner._preprocess()`.

When vLLM runs a padded CUDA graph shape, `num_input_tokens` can be larger than
`scheduler_output.total_num_scheduled_tokens`. Before this PR, `_preprocess()`
zeroed the padded tail of `positions` for the plain text path, but returned
text-only `input_ids = self.input_ids.gpu[:num_input_tokens]` without clearing
the padded tail. Those rows could therefore contain stale token IDs from a
previous scheduler step.

This PR makes padded model inputs deterministic by zeroing the padded tail for:

- `input_ids`, when the model receives token IDs
- `inputs_embeds`, when the model receives embeddings
- `positions`, for 1-D and multi-axis position tensors

The fix preserves the real scheduled rows and only touches
`[num_scheduled_tokens:num_input_tokens]`.

## Test Plan

Added a focused unit regression in `tests/v1/worker/test_gpu_model_runner.py`.
The test constructs a minimal text-only `GPUModelRunner` object and calls
`_preprocess()` with:

- `total_num_scheduled_tokens = 3`
- `num_input_tokens = 5`
- real input rows: `[11, 12, 13]`
- stale padded input rows: `[991, 992]`

It verifies that the real rows are preserved and the padded `input_ids` and
`positions` rows are zeroed.

Command:

```bash
uv run pytest -q tests/v1/worker/test_gpu_model_runner.py -k padded_model_inputs
```

## Test Result

Before the fix, the new regression failed because padded `input_ids` retained
the stale values:

```text
assert torch.equal(input_ids[3:], torch.zeros(2, dtype=torch.int32))
E       assert False
E        +  where False = torch.equal(
E              tensor([991, 992], dtype=torch.int32),
E              tensor([0, 0], dtype=torch.int32),
E          )
```

After the fix:

```text
1 passed, 32 deselected
```

The commit hooks also passed during `git commit -s`, including ruff, format,
SPDX checks, forbidden import checks, and signoff validation.
