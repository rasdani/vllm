# Zero padded model inputs before execution

## Purpose

Fix stale padded model inputs in V1 `GPUModelRunner._preprocess()`.

When vLLM runs a padded CUDA graph shape, `num_input_tokens` can be larger than
`scheduler_output.total_num_scheduled_tokens`. The real scheduled rows are
refreshed before the forward pass, but the padded tail can still contain values
left by an earlier scheduler step.

Existing cleanup from #37873 zeroed stale padded `positions` for the plain
position path. This PR extends the same padded-tail cleanup to the full model
input surface passed into the padded forward:

- `input_ids`, when the model receives token IDs
- `inputs_embeds`, when the model receives embeddings
- `positions`, for 1-D and multi-axis position tensors

The fix preserves the real scheduled rows and only touches
`[num_scheduled_tokens:num_input_tokens]`.

Related context:

- #37873 fixed stale CUDA graph padding `positions` in `_preprocess`.
- #16337 discussed CUDA graph padding executing models with padded inputs beyond
  `total_num_scheduled_tokens`, but was closed stale without a targeted fix for
  stale `input_ids`.

## Duplicate-work check

I checked for related upstream work before opening this PR:

```bash
gh issue view 16337 --repo vllm-project/vllm --comments
gh pr list --repo vllm-project/vllm --state open --search "16337 in:body"
gh pr list --repo vllm-project/vllm --state open --search "stale padded model inputs GPUModelRunner _preprocess"
gh pr list --repo vllm-project/vllm --state open --search "CUDA graph padding input_ids"
```

Results:

- No open PRs reference #16337.
- No open PRs matched the targeted stale padded model input search.
- The broad `CUDA graph padding input_ids` search only found unrelated SM121/GB10
  enablement work.

This is not duplicating #37873: that PR fixed stale padded `positions`, while
this PR also clears stale padded `input_ids` and `inputs_embeds` and handles
multi-axis position tensors through the shared `positions[..., pad_slice]` path.

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

Command to run:

```bash
uv run pytest -q tests/v1/worker/test_gpu_model_runner.py -k padded_model_inputs
```

## Test Result

Not run locally in this session, per maintainer request.

Expected failure mode before this fix: the new regression fails because padded
`input_ids` retain stale values such as `[991, 992]` instead of zeros.

## AI assistance

AI assistance was used to draft and implement this change. The submitting human
should review every changed line and run the relevant tests before marking this
draft ready for review.

---
<details>
<summary> Essential Elements of an Effective PR Description Checklist </summary>

- [x] The purpose of the PR, such as "Fix some issue (link existing issues this PR will resolve)".
- [x] The test plan, such as providing test command.
- [x] The test results, such as pasting the results comparison before and after, or e2e results.
- [ ] (Optional) The necessary documentation update, such as updating `supported_models.md` and `examples` for a new model.
- [ ] (Optional) Release notes update. If your change is user facing, please update the release notes draft in the Google Doc.

</details>
