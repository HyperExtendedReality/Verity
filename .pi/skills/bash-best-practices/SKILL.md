---
name: bash-best-practices
description: "Reusable procedure and error prevention for bash"
---

# bash Best Practices and Error Recovery
> Command timed out after 30 seconds
> Diff in \\?\C:\Dev\Game Dev\Verity\crates\verity-lux\tests\split_projection.rs:60:
> float nominalCosine = 0.0;

## Failing invocation(s)
- `bash` called with {"command":"rustfmt --edition 2024 --check crates/verity-lux/tests/split_projection.rs","timeout":20}
- `bash` called with {"command":"git --no-pager diff --no-index --unified=2 -- 'C:/Users/hyper/AppData/Local/Temp/verity-split-pack-probes-9brfjngp/split.slang' 'C:/Dev/Game Dev/Ver

## Recommended Procedure
1. Verify command syntax, paths, and tool constraints prior to execution.
2. Avoid speculative flags or unverified assumptions.
3. Inspect intermediate outputs before chained operations.