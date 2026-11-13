// Matrix record layouts, read through the pointers in `Globals`. Here rather
// than in matrix.slang so common.slang can name them. Layouts mirror `matrix.rs`.

struct MatrixInstance {
    Instance base;
    float4x4 inverse;
    /// World-space bounds: center, radius.
    float4 sphere;
    /// x: asset, y: 1 when two-sided, z: largest axis scale as float bits.
    uint4 info;
};

struct MatrixAsset {
    /// Object-space bounds: center, radius.
    float4 sphere;
    uint first_root;
    uint root_count;
    /// The imposter's bindless textures, or NO_TEXTURE.
    uint imposter_albedo;
    uint imposter_surface;
    uint first_cluster;
    uint cluster_count;
    uint2 padding;
};

struct MatrixNode {
    float4 sphere;
    float error;
    uint group;
    uint first_child;
    uint child_count;
};

struct MatrixGroup {
    float4 sphere;
    float error;
    uint first_cluster;
    uint cluster_count;
    uint padding;
};

struct MatrixCluster {
    float4 sphere;
    uint refined;
    uint first_index;
    uint triangle_count;
    uint first_brick;
    uint brick_count;
    uint depth;
    uint2 padding;
};

struct MatrixBrick {
    float3 origin;
    float voxel_size;
    uint2 mask;
    uint first_voxel;
    uint padding;
};
