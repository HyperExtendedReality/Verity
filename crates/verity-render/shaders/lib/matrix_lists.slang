// Where selection's draw lists live in `matrix_entries`, and the layout of
// `matrix_counters`. Mirrors `matrix.rs`.

static const uint MATRIX_LIST_ONE_SIDED = 0;
static const uint MATRIX_LIST_TWO_SIDED = 1;
static const uint MATRIX_LIST_VOXELS = 2;
static const uint MATRIX_LIST_IMPOSTER = 3;
static const uint MATRIX_LIST_SHADOW = 4;
static const uint MATRIX_LIST_SHADOW_VOXELS = 5;
/// Words before the per-pass dispatch arguments: one draw command per list.
static const uint MATRIX_PASS_WORDS = 6 * 4;
/// Items in each half of the queue.
static const uint MATRIX_QUEUE = 1u << 20;

static const uint MATRIX_CLUSTER_LIST = 1u << 19;
static const uint MATRIX_BRICK_LIST = 1u << 21;

uint matrix_list_capacity(uint list) {
    return list == MATRIX_LIST_VOXELS || list == MATRIX_LIST_SHADOW_VOXELS ? MATRIX_BRICK_LIST : MATRIX_CLUSTER_LIST;
}

uint matrix_list_base(uint list) {
    uint base = 0;
    for (uint i = 0; i < list; ++i)
        base += matrix_list_capacity(i);
    return base;
}
