// Where selection's draw lists live in `matrix_entries`, and the layout of
// `matrix_counters`. Mirrors `matrix.rs`.

static const uint MATRIX_LIST_ONE_SIDED = 0;
static const uint MATRIX_LIST_TWO_SIDED = 1;
static const uint MATRIX_LIST_VOXELS = 2;
static const uint MATRIX_LIST_IMPOSTER = 3;
/// Then each shadow layer's clusters and bricks: sun cascades, then point faces.
static const uint MATRIX_LIST_LAYERS = 4;
static const uint MATRIX_SHADOW_LAYERS = CASCADES + MAX_POINT_SHADOWS * POINT_SHADOW_FACES;
static const uint MATRIX_LISTS = MATRIX_LIST_LAYERS + MATRIX_SHADOW_LAYERS * 2;
/// Words before the per-pass dispatch arguments: one draw command per list.
static const uint MATRIX_PASS_WORDS = MATRIX_LISTS * 4;
/// Items in each half of the queue.
static const uint MATRIX_QUEUE = 1u << 20;

static const uint MATRIX_CLUSTER_LIST = 1u << 19;
static const uint MATRIX_BRICK_LIST = 1u << 21;
/// A cascade sees part of the scene, a point face far less.
static const uint MATRIX_CASCADE_CLUSTERS = 1u << 17;
static const uint MATRIX_CASCADE_BRICKS = 1u << 19;
static const uint MATRIX_FACE_CLUSTERS = 1u << 15;
static const uint MATRIX_FACE_BRICKS = 1u << 16;

uint matrix_list_capacity(uint list) {
    if (list < MATRIX_LIST_LAYERS)
        return list == MATRIX_LIST_VOXELS ? MATRIX_BRICK_LIST : MATRIX_CLUSTER_LIST;
    bool bricks = (list & 1) != 0;
    if ((list - MATRIX_LIST_LAYERS) / 2 < CASCADES)
        return bricks ? MATRIX_CASCADE_BRICKS : MATRIX_CASCADE_CLUSTERS;
    return bricks ? MATRIX_FACE_BRICKS : MATRIX_FACE_CLUSTERS;
}

uint matrix_list_base(uint list) {
    uint main = 3 * MATRIX_CLUSTER_LIST + MATRIX_BRICK_LIST;
    if (list < MATRIX_LIST_LAYERS)
        return list <= MATRIX_LIST_VOXELS ? list * MATRIX_CLUSTER_LIST : main - MATRIX_CLUSTER_LIST;
    uint layer = (list - MATRIX_LIST_LAYERS) / 2;
    uint bricks = list & 1;
    if (layer < CASCADES)
        return main + layer * (MATRIX_CASCADE_CLUSTERS + MATRIX_CASCADE_BRICKS) + bricks * MATRIX_CASCADE_CLUSTERS;
    return main + CASCADES * (MATRIX_CASCADE_CLUSTERS + MATRIX_CASCADE_BRICKS)
        + (layer - CASCADES) * (MATRIX_FACE_CLUSTERS + MATRIX_FACE_BRICKS) + bricks * MATRIX_FACE_CLUSTERS;
}
