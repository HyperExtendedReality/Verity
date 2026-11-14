// Wave-batched hash lookups, shared by Split's world cache and surface cells
// and PXTGI's cache. Neighbouring lanes (texels of one island, pixels of one
// tile, the corners of nearby samples) often look up the same key: the wave's
// first active lane probes once, and lanes with its key and tag take its slot
// instead of each reloading the probe sequence and touching the table's
// atomics. Lanes with other keys probe on their own. The tag holds whatever
// else must match, such as whether the lookup may claim a slot.
//
//     bool follows = waveBatchFollows(key, tag);
//     uint slot = follows ? EMPTY : probe();
//     bool taken;
//     uint first = waveBatchTake(slot, key, tag, follows, taken);
//     if (taken) slot = first; else if (follows) slot = probe();

// Whether this lane can take the first active lane's lookup.
bool waveBatchFollows(uint key, uint tag) {
    return !WaveIsFirstLane() && all(WaveReadLaneFirst(uint2(key, tag)) == uint2(key, tag));
}

// The first active lane's slot. `taken` is false unless this lane follows and
// that lane still is the one that probed for the same key and tag, so lanes
// that did not reconverge between the two calls probe for themselves.
uint waveBatchTake(uint slot, uint key, uint tag, bool follows, out bool taken) {
    uint4 first = WaveReadLaneFirst(uint4(slot, key, tag, follows ? 0u : 1u));
    taken = follows && first.w == 1u && first.y == key && first.z == tag;
    return first.x;
}
