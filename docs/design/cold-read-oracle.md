# Design: in-blob routing region for page-granular cold reads

Status: **proposed (design)**. Breaking blob-layout change (manifest +1 version).
Replaces the external `cold.idx` sidecar. Builds on `read_blob_range`
(committed `808a5fa`).

## Problem (measured)

A point lookup that misses the buffer pool pins the **whole 512 KB blob frame**
to answer one key. Measured: `bm_read_bytes / bm_point_reads ≈ 529 KB` per cold
read — ~128× read amplification (512 KB I/O for a ~100 B answer). This is why
holt's cold reads lose to RocksDB (which reads one ~4 KB SST block + bloom).

The `cold.idx` sidecar "fixed" this by caching `(key→value)` in a second,
**unbounded, unaccounted** in-RAM table — a hit-rate play that (a) is no better
than enlarging the buffer pool, (b) can't help when the working set >> RAM, and
(c) introduced a class of crash/staleness bugs (separate mutable sidecar;
generation aliasing across crash; torn-tail; sidecar I/O failing user reads).
See the cold.idx safety review.

The fundamental lever is **miss cost**, not hit rate: stop reading 512 KB to
answer one key.

## Measured ceiling (`cold.rs::cold_read_page_touch_ceiling`)

objstore 300k keys / 48 B values / 225 blobs (~1333 keys/blob): a point-lookup
descent touches **mean 4.64 distinct 4 KB pages (~18.6 KB), p95 24 KB** — vs the
512 KB pin. R1 already keeps the descent off the 40 KB slot table (pages 1–10);
it touches the header page + scattered data-area node pages + the leaf page. So
even naive "pread the touched pages" is ~27× less cold I/O. Clustering can push
this to ~1–2 pages.

structure/value split = **78% / 22%** at 48 B values → "keep all *structure*
resident" is NOT universal (for small values the structure *is* the data). The
**routing nodes alone** (internals, no keys/values) are a small fraction — that
is what this design makes contiguous + cheaply readable.

## Design: routing region + targeted leaf read

A breaking blob layout, built at compaction/spillover (when the blob is
rewritten and all its keys are known), that lets a cold read reuse the **existing
descent** without reimplementing it, and without a separate sidecar.

### Layout (per 512 KB blob)

```
[ 0x0000          ] BlobHeader (4 KB, page 0)   — gains routing-region fields
[ 0x1000..0xB000  ] slot table (40 KB)          — unchanged (off the read path since R1)
[ 0xB000..R_end   ] ROUTING REGION              — ALL internal nodes, contiguous:
                                                   root, Prefix, Node4/16/48/256, BlobNode
[ R_end..0x80000  ] LEAF REGION (page-aligned)  — leaves [16B hdr][key][value]
```

New `BlobHeader` fields (reuse `_pad`): `routing_off: u32`, `routing_len: u32`,
`leaf_region_start: u32`. Legacy blobs (pre-format) have these `0` → cold path
falls back to a full pin (so old data just doesn't get the optimization).

**Invariant (build-enforced):** every offset `< leaf_region_start` is an
internal node; every offset `>= leaf_region_start` is a leaf. So the cold
descent can tell "internal vs leaf" from the offset *without* reading the node.

### Build (compaction / spillover, in `migrate.rs` / `spillover.rs`)

`clone_subtree` already DFS-walks the source blob in key order. Change it to a
**two-arena writer**: internal nodes → routing arena (front), leaves → leaf
arena (page-aligned, after). Child offsets are back-patched as today (R1
offset-addressing is unchanged; offsets just land in two contiguous zones).
Write the three header fields. No new index structure, no MPH, no bloom build —
the routing nodes *are* the index.

Cost: a second cursor in the existing compaction walk; same number of bytes
written. The routing region is naturally small (internals only).

### Cold read path (`cold_lookup_or_pin` fallback → new `cold_read_routed`)

When the sidecar/BM says a blob is non-resident and `routing_len != 0`:

1. Load the routing region: `read_blob_range(guid, routing_off, &mut buf[..routing_len_pages])`
   — 1–2 pages (contiguous), OR served from a small resident routing cache
   (see "resident routing" below).
2. Wrap `[header ++ routing region]` and run the **existing descent**
   (`resolve_typed`/`child_offset`/`lookup_at`). Every internal node is present.
3. When the descent reaches a child whose offset `>= leaf_region_start`, it is a
   leaf: `read_blob_range(guid, page_of(off), &mut leaf_page)` (one 4 KB page,
   or two if the leaf straddles a boundary) → read `[16B hdr][key][value]` →
   compare the full key (with terminator) → return `Found{value,seq}` /
   `NotFound`.
4. A `BlobNode` (crossing) resolves in the routing region → recurse into the
   child blob's routing region (the loop already in `cold_lookup_or_pin`).

The descent is reused verbatim except the **single leaf-access step** becomes a
targeted `read_blob_range` instead of an inline slice deref. That localized
change is the only new descent code — far less risk than a from-scratch paged
descent (it cannot diverge on the routing logic, only on the leaf read, which a
`routed == full-pin` correctness test pins exactly).

Cost: **2–3 preads (~8–12 KB)** per cold positive read; **1 pread** if routing is
resident; vs 512 KB. ~40–128× less cold I/O, value-size-agnostic, zero
steady-state value cache.

### Resident routing (optional, the big win)

Routing regions are small (internals only). Keep them resident in a **bounded,
accounted** small cache (or fold into the BM as a distinct class): ~a few KB ×
N_blobs (≈ 15–30 MB for 5 M keys — vs cold.idx's 1 GB+). Then a cold read is a
single 4 KB leaf pread. This is the honest "keep the *index* resident" — bounded,
value-agnostic, and 1/30th the RAM of caching values.

### Negatives (compose with R2 bloom, later)

The routing descent already answers most negatives cheaply (a missing byte at an
internal node → `NotFound`, no leaf read). A per-blob bloom in the header would
make *within-prefix* negatives free; orthogonal, additive, later.

## Why this is crash-safe by construction (fixes the cold.idx review)

The routing region lives **inside the blob** and is written **atomically with
it** (one blob write at compaction). There is:
- **no separate mutable sidecar** → no generation-aliasing-across-crash, no
  torn-`cold.idx`, no sidecar-I/O-fails-the-user-read;
- **no staleness** → the routing region always matches the blob image it is part
  of (like R3 inlining the leaf, this inlines the index);
- **no generation check needed** → it's the same bytes, same image, same
  manifest entry.

Recovery is unchanged: a blob is either the old image (old routing region) or the
new (atomic). The cold path reads whatever the manifest points at.

## Staged, independently-testable plan

1. **Header fields + reader** (no behavior change): add `routing_off/len/
   leaf_region_start`, default 0; bump manifest blob-format version; legacy
   blobs read as "no routing" → full-pin fallback. Validate: existing suite
   unchanged; old data opens.
2. **Two-arena compaction build**: `clone_subtree` writes internals→routing,
   leaves→leaf region; set header fields. Validate: a `routing == full` invariant
   test (descend the routing region + leaf region, assert identical key set +
   values to a full-frame descent) over a built blob; proptest vs BTreeMap.
3. **Cold routed read**: wire `cold_read_routed` into the `cold_lookup_or_pin`
   non-resident fallback (replacing the full pin), using `read_blob_range`.
   Validate: **`routed_get(key) == tree.get(key)` for 100k random keys (present,
   absent, crossing)** — the data-integrity gate; dual-arch; bench cold
   `bm_read_bytes` drop (~512 KB → ~8–12 KB).
4. **Resident routing cache** (bounded, accounted): keep routing regions hot;
   cold read → 1 leaf pread. Validate: RAM bound respected; cold A/B vs RocksDB
   at matched memory.
5. **Remove `cold.idx`**: the routing region subsumes it; delete the sidecar +
   its review-flagged hazards. Validate: full suite + SIGKILL crash-soak
   (`wal_crash_soak`) green without the sidecar.
6. (later) **Per-blob bloom in the header** for free within-prefix negatives.

## Honest expectations / open questions

- Positive cold read: 512 KB → ~8–12 KB (on-disk routing) or ~4 KB (resident
  routing). Negative: routing descent or bloom. Value-size agnostic.
- Routing-region size variance: dense small-value blobs have more internals;
  confirm routing_len stays ≤ ~2–3 pages (measure during stage 2). If a blob's
  internals exceed the budget, fall back to full pin for that blob.
- Leaf straddling a page boundary → up to 2 leaf preads; values > 4 KB span more
  pages (rare; the leaf read already knows value_len → reads exactly the span).
- Compaction-time cost of the two-arena write (should be ~neutral; measure).
- Interaction with snapshots/CoW: a snapshot reads an older blob image via its
  own manifest entry → its routing region is that image's; no special case.
