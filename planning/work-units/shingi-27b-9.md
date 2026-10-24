# Smarty inference checkout cleanup

Work order: https://github.com/kortexa-ai/shingi-27b/issues/9

The user requested cleanup of completed experiment checkouts, preservation of
branch-only work, and commits for dirty source. The comparison base is this
inference repository at `78c9f5e1168cfef202f2df55fda44c6661ce6195`. The separate
`kortexa-ai/shingi` training repository is not the integration target.

All candidates are standalone clones. Squashed integration requires content
comparison in addition to ancestry checks:

| Checkout | Evidence | Disposition |
| --- | --- | --- |
| `shingi-parallel-20261008` | `7f43e08` runtime and tests match integrated `b2e1393`; main adds report data and the baseline-aware benchmark correction | Archive and remove |
| `shingi-throughput-20261008` | `218f881` differs from integrated `2f7acfd` only by the added report | Archive and remove |
| `shingi-vram-cache-20261008` | `7b5f7c7` differs from integrated `efdca28` only by README and report additions | Archive and remove |
| `shingi-repin-20261022` | `kortexa-ai/prism-repin-4fbda129` at `82c419e` has four commits beyond main | Keep |
| `shingi-sequence-scaling-20261008` | Checked-out `dafeb38` is merged; local sequence and kernel branches retain experiment history, including the unpromoted native profiler controls at `677fcb3` | Keep |
| `shingi-kernel-runtime-20261009` | Prism base `d8f26ee` has four modified source files and compiled experiment artifacts | Commit existing source on a preservation branch and keep |

The sequence branch's native source SHA-256 is
`2e79eede5a76455e44c497129c2ae3f6c8f859a244cc47f4334aaa3afbae8913`, matching the
reproduction hash already published in `results/sequence-scaling/REPORT.md`.
The kernel investigation branch is integrated with added reports at `dafeb38`.

Full archives of removed clones preserve Git history and ignored research
artifacts. Kept experiment refs are also saved as Git bundles. Recovery files and
checksums are on Smarty under
`~/Desktop/ai reports/smarty-checkout-cleanup-2026-10-24/shingi-27b/`.

No production package, runtime pin, process, or recipe changes. Source preservation
does not establish correctness or a speed improvement for the old kernel trials.

On 2026-10-24 UTC, the three completed clones were removed after full archive
comparison and a second clean-status, HEAD and consumer check. All 16 running
Kortexa service PIDs, start times and states were unchanged. Shingi health
returned HTTP 200 with `status: ok`.

The four existing Prism source edits are committed as
`6d6cebdb4038c7ca8a1d0d4efe26b5b54ee9ddaa` on local branch
`kortexa-ai/preserve-shingi-kernel-trials-20261024`. The before/after source
hashes are identical; `git diff --check` passed. The compiled files remain in
place and are excluded locally through `.git/info/exclude`, not added to Git.
All kept checkouts are clean. The three retained experiment clones have verified
Git bundles; no unreviewed experiment was merged into main or deployed.

Related model checkout cleanup: https://github.com/kortexa-ai/models.server/issues/83.
