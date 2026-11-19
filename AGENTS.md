# Agent notes

## Box3D is not part of Verity

Verity's physics is `verity-physics`. Box3D (`crates/box3d`, `crates/verity-box3d`)
exists only as a reference to compare `verity-physics` performance and accuracy
against.

- Never use Box3D for engine features, and never add `verity-box3d` as a
  dependency of another crate or as a normal dependency of `verity-physics`.
- Never build physics, collision or anything else on top of it. If
  `verity-physics` lacks something, add it to `verity-physics`.
- Don't build it for work that has nothing to do with physics, and don't set
  `BOX3D_SOURCE_DIR` to get past a build error. Leave it out instead:

```bash
cargo test --workspace --exclude verity-box3d --exclude verity-physics
```

```bash
cargo clippy --workspace --exclude verity-box3d --exclude verity-physics
```

`verity-physics` itself builds without Box3D. Its tests and benchmarks are the one
place Box3D is needed, because it is their dev-dependency, so only touch it when
you are changing or measuring `verity-physics`.

<!-- pi-harness2:rules:start -->
- When encountering complex or repeated operations with `bash`, load the `bash-best-practices` skill.
<!-- pi-harness2:rules:end -->
