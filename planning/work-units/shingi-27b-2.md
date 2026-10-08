# Parallel decision inference

Work order: https://github.com/kortexa-ai/shingi-27b/issues/2

Use Prism's existing sequence and shared-prefix operations with one model allocation.
Keep an explicit serial comparison path, preserve candidate-logit semantics, and bound
both admitted requests and the shared context token budget. The Python transport owns
one native pipe and routes each result to its original caller.

Validate transport failure/shutdown and API compatibility without a GPU, then run frozen
serial/parallel comparisons, cache isolation, malformed-input recovery, image and long
context checks, repeated throughput measurements and sampled memory checks. Keep exact
inputs and raw outputs in ignored artifacts and publish compact reproducible evidence.

Development stays on an unpublished local branch. Transfer candidate commits for GPU
checks through Git bundles. Consolidate the validated implementation into one commit on
main and push only main. The owning issue records live status and validation gates.
