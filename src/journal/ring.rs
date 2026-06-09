//! Lock-free shared WAL ring (see `docs/design/wal-ring.md`).
//!
//! The append substrate for [`super::group_commit::Journal`]. Replaced the
//! per-record `Vec` + single crossbeam channel + single batching worker (the
//! measured concurrent-write bottleneck: durable writes capped at ~0.29
//! Mops/s @16t while the in-memory ART path scaled to 5.78). N writers append
//! to ONE ordered log concurrently:
//!
//! 1. `reserve(total_len)` — one `fetch_add` on `tail`, the byte cursor.
//!    Successive reservations tile `[0, tail)` with NO gaps (each start ==
//!    the previous end), so the byte address IS the dense, gap-free order
//!    key — there is no second counter to disagree with it.
//! 2. `fill(ticket, bytes)` — parallel, non-atomic memcpy of the
//!    already-encoded record into the writer's disjoint byte range.
//! 3. `publish(ticket)` — under a brief `advance` lock, record the
//!    published byte interval and greedily fold the contiguous published
//!    prefix into `committed_addr`.
//! 4. A single flusher copies `[flush_cursor, committed_addr)` — a true
//!    contiguous, fully-written, in-order prefix — into the (unchanged)
//!    WAL writer.
//!
//! ## Why byte-keyed, not work-id-keyed
//!
//! An earlier design kept a separate dense `work_alloc` counter as the
//! order key. The loom model below caught the flaw (the design doc flagged
//! it as an open question): `work_alloc` and `tail` are independent
//! `fetch_add`s whose orders can DISAGREE — writer A can get work-id 1 but
//! byte range `[4,8)` while writer B gets work-id 2 but `[0,4)`. Folding by
//! work-id then advances `committed_addr` over a byte range whose lower
//! bytes are not yet published → the flusher copies an unpublished gap
//! (silent corruption; CRC would NOT catch it — each record is
//! individually valid). Keying on the byte tiling uses the one natural
//! order and is immune.
//!
//! ## Why the `advance` lock is load-bearing (not just mutual exclusion)
//!
//! It chains every writer's memcpy into the watermark publication. A writer
//! does its plain-store memcpy, THEN locks `advance` to record its
//! interval. Whichever writer later folds the interval starting at
//! `committed_addr` does so only after acquiring `advance` — which
//! synchronizes-with the unlock of the writer that filled it, so that
//! writer's memcpy *happens-before* the `committed_addr` Release store. The
//! flusher's single Acquire load of `committed_addr` therefore observes
//! every byte in `[0, committed_addr)`. No torn / gap copy is possible.
//!
//! ## W2D + no-stall (preserved from the work-id design)
//!
//! - W2D: checkpoint captures `committed_addr` (via the Journal's record-count
//!   watermark) under `commit_gate`'s exclusive side, which waits for in-gate
//!   writers; `publish` folds the writer's interval synchronously before it
//!   releases the gate, so the captured watermark covers every record whose
//!   blob is in the dirty snapshot.
//! - No prefix stall on `next_seq` gaps: the byte range is allocated in
//!   `reserve`, at the point a record actually exists — failed guards /
//!   early returns burn `next_seq`, never `tail`.
//!
//! The ring is a pure append substrate: `copy_committed_prefix` drains into a
//! caller-supplied sink. `group_commit` owns the `WalWriter`, the sync-ack
//! path, backpressure parking, and checkpoint integration.

// `group_commit::Journal` is the production consumer; a few small accessors
// (`append`, `tail`, …) exist only for the ring's own unit/loom tests.
#![allow(dead_code)]
// `loom` is a build-time cfg set only by the model-check pass
// (`RUSTFLAGS="--cfg loom"`), never a Cargo feature, so the lint can't see it.
#![allow(unexpected_cfgs)]

use std::collections::BTreeMap;

// loom swaps in its model-checked atomics/Mutex under `--cfg loom`; normal
// builds and tests use std. The ring's logic is identical either way.
#[cfg(loom)]
use loom::sync::{
    atomic::{AtomicU64, Ordering},
    Mutex,
};
#[cfg(not(loom))]
use std::sync::{
    atomic::{AtomicU64, Ordering},
    Mutex,
};

/// A reservation handed out by [`WalRing::reserve`], consumed by
/// [`WalRing::fill`] + [`WalRing::publish`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ReserveTicket {
    /// Logical start byte address (monotone; physical = `start & mask`).
    pub(crate) start: u64,
    /// Logical end byte address (`start + total_len`). Also the dense,
    /// gap-free order key: the next reservation starts exactly here.
    pub(crate) end: u64,
}

impl ReserveTicket {
    #[inline]
    fn len(self) -> usize {
        (self.end - self.start) as usize
    }
}

/// Published-but-not-yet-folded byte intervals, keyed by start address.
#[derive(Debug)]
struct Advancer {
    /// `start_addr -> end_addr`. Bounded by the number of in-flight
    /// reservations. Folded greedily from `committed_addr`.
    pending: BTreeMap<u64, u64>,
    /// Count of records folded into the committed prefix (dense, monotone).
    committed_count: u64,
}

/// Fixed-size in-RAM byte ring. Writers reserve disjoint, gap-free byte
/// ranges and memcpy in parallel; a single flusher drains the committed
/// contiguous prefix in byte (== reservation == file) order.
pub(crate) struct WalRing {
    /// Backing bytes. `UnsafeCell` so disjoint reserved ranges can be
    /// written concurrently without a data race (ranges never overlap; the
    /// flusher reads only published bytes via the `advance`-lock HB chain).
    buf: Box<[std::cell::UnsafeCell<u8>]>,
    /// `capacity - 1`; capacity is a power of two so `& mask` wraps.
    mask: u64,
    capacity: u64,

    /// Next free LOGICAL byte address (monotone u64, never wraps).
    tail: AtomicU64,
    /// Highest byte address `A` such that `[0, A)` is fully filled +
    /// published (a contiguous run of reserved intervals). Advanced only
    /// under `advance`, with Release ordering. The flusher's Acquire load of
    /// this is the sole synchronization edge to the bytes.
    committed_addr: AtomicU64,
    /// Count of records in the committed prefix (mirrors
    /// `Advancer::committed_count` for lock-free reads).
    committed_records: AtomicU64,

    /// Logical byte address already copied out by the flusher. Flusher
    /// writes (Release); writers read (Acquire) for backpressure.
    flush_cursor: AtomicU64,

    /// Serializes prefix folding. Held only briefly (BTreeMap insert + a
    /// short contiguous fold); see the module HB argument.
    advance: Mutex<Advancer>,
}

// SAFETY: the only shared mutable access to `buf` is (a) writers memcpy-ing
// into disjoint byte ranges they exclusively reserved via `tail.fetch_add`,
// and (b) the flusher reading bytes strictly below `committed_addr`, whose
// Acquire load happens-after every contributing writer's memcpy (via the
// `advance` lock release/acquire chain). No two accesses alias a live byte.
unsafe impl Sync for WalRing {}
unsafe impl Send for WalRing {}

impl WalRing {
    /// Create a ring with the given capacity in bytes (rounded up to a power
    /// of two, min 64). A record must fit in `capacity` (oversize fallback
    /// is stage 5).
    pub(crate) fn with_capacity(capacity_bytes: usize) -> Self {
        let cap = capacity_bytes.max(64).next_power_of_two();
        let buf = (0..cap)
            .map(|_| std::cell::UnsafeCell::new(0u8))
            .collect::<Vec<_>>()
            .into_boxed_slice();
        WalRing {
            buf,
            mask: (cap as u64) - 1,
            capacity: cap as u64,
            tail: AtomicU64::new(0),
            committed_addr: AtomicU64::new(0),
            committed_records: AtomicU64::new(0),
            flush_cursor: AtomicU64::new(0),
            advance: Mutex::new(Advancer {
                pending: BTreeMap::new(),
                committed_count: 0,
            }),
        }
    }

    #[inline]
    pub(crate) fn capacity(&self) -> u64 {
        self.capacity
    }

    /// Reserve a `total_len`-byte slot. One `fetch_add` on `tail`; the
    /// returned range is contiguous with its predecessor (gap-free tiling).
    #[inline]
    pub(crate) fn reserve(&self, total_len: u64) -> ReserveTicket {
        debug_assert!(total_len > 0 && total_len <= self.capacity);
        let start = self.tail.fetch_add(total_len, Ordering::Relaxed);
        ReserveTicket {
            start,
            end: start + total_len,
        }
    }

    /// True once the flusher has freed enough RAM for `ticket`'s range to be
    /// safely overwritten (gates on `flush_cursor`, decoupled from fsync).
    /// Stage 5 turns the spin into parking; stage 1 callers size the ring so
    /// this never trips.
    #[inline]
    pub(crate) fn reserve_space_ready(&self, ticket: &ReserveTicket) -> bool {
        ticket.end <= self.flush_cursor.load(Ordering::Acquire) + self.capacity
    }

    /// memcpy the encoded record into the reserved range (wrap-split). Plain
    /// stores; the happens-before edge is established by `publish`.
    pub(crate) fn fill(&self, ticket: &ReserveTicket, bytes: &[u8]) {
        debug_assert_eq!(bytes.len(), ticket.len());
        debug_assert!(
            self.reserve_space_ready(ticket),
            "ring overrun (stage 5: backpressure)"
        );
        let cap = self.buf.len();
        let off = (ticket.start & self.mask) as usize;
        let first = bytes.len().min(cap - off);
        // SAFETY: `[off, off+first)` and the wrapped `[0, rest)` lie inside
        // the writer's exclusively reserved, non-overlapping range.
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), self.buf[off].get(), first);
            if first < bytes.len() {
                std::ptr::copy_nonoverlapping(
                    bytes[first..].as_ptr(),
                    self.buf[0].get(),
                    bytes.len() - first,
                );
            }
        }
    }

    /// Record `ticket`'s published interval and greedily fold the contiguous
    /// published prefix into `committed_addr`. MUST be called after `fill`
    /// (the memcpy happens-before this lock acquisition).
    pub(crate) fn publish(&self, ticket: &ReserveTicket) {
        let mut adv = self.advance.lock().unwrap();
        adv.pending.insert(ticket.start, ticket.end);
        // Fold every contiguous published interval starting at committed_addr.
        // Relaxed load is sufficient under the lock (only this critical
        // section advances committed_addr); the flusher reads it with Acquire.
        let mut addr = self.committed_addr.load(Ordering::Relaxed);
        let start_addr = addr;
        let mut folded = 0u64;
        while let Some(end) = adv.pending.remove(&addr) {
            addr = end;
            folded += 1;
        }
        if addr != start_addr {
            adv.committed_count += folded;
            // Store addr BEFORE records (both Release): this guarantees
            // "committed_records visible ⟹ committed_addr visible". A flusher
            // that reads committed_records == RC (Acquire) and then loads
            // committed_addr is guaranteed CA >= end(RC), so copying to CA
            // drains >= RC records and `flushed = base + RC` is a safe lower
            // bound on durable records.
            self.committed_addr.store(addr, Ordering::Release);
            self.committed_records
                .store(adv.committed_count, Ordering::Release);
        }
        // `adv` unlock = Release: this writer's memcpy (done before the lock)
        // happens-before any later folder's Acquire of `advance`, hence
        // before the flusher's Acquire of committed_addr.
    }

    /// Copy the committed contiguous prefix `[flush_cursor, committed_addr)`
    /// into `sink` (once per contiguous physical run — twice on wrap).
    /// Advances `flush_cursor`. Returns bytes copied. Single-flusher only.
    pub(crate) fn copy_committed_prefix(&self, sink: &mut impl FnMut(&[u8])) -> u64 {
        // Acquire: synchronizes-with the publishing fold's Release store,
        // making every byte in [0, committed_addr) visible here.
        let committed = self.committed_addr.load(Ordering::Acquire);
        let from = self.flush_cursor.load(Ordering::Acquire);
        if committed <= from {
            return 0;
        }
        let total = (committed - from) as usize;
        let cap = self.buf.len();
        let off = (from & self.mask) as usize;
        let first = total.min(cap - off);
        // SAFETY: all bytes in [from, committed) were filled by writers whose
        // memcpy happens-before this Acquire load (advance-lock chain).
        // committed_addr only advances over fully-published ranges, so no
        // writer is still filling here.
        unsafe {
            sink(std::slice::from_raw_parts(self.buf[off].get(), first));
            if first < total {
                sink(std::slice::from_raw_parts(self.buf[0].get(), total - first));
            }
        }
        self.flush_cursor.store(committed, Ordering::Release);
        total as u64
    }

    /// Reset the byte cursors to 0 after the ring has been fully drained
    /// (post-checkpoint truncate, when the on-disk WAL is reset to its
    /// header). The record count is deliberately NOT reset — it is a stable
    /// global order across truncations (mirrors today's never-reset work id).
    ///
    /// Caller must guarantee no concurrent `reserve`/`fill`/`publish`/`copy`
    /// (the checkpoint truncate boundary holds `commit_gate` exclusively and
    /// the flusher is caught up). Asserts the ring is fully published +
    /// drained.
    pub(crate) fn reset_after_drain(&self) {
        let adv = self.advance.lock().unwrap();
        debug_assert!(adv.pending.is_empty(), "unpublished intents at reset");
        let committed = self.committed_addr.load(Ordering::Relaxed);
        debug_assert_eq!(self.tail.load(Ordering::Relaxed), committed, "tail not published");
        debug_assert_eq!(
            self.flush_cursor.load(Ordering::Relaxed),
            committed,
            "flusher not caught up"
        );
        // Order: cursors to 0 while holding `advance` so no folder/flusher
        // observes a torn (committed_addr=0, tail=old) state. Release so a
        // subsequent reserve/copy on another thread sees the reset.
        self.flush_cursor.store(0, Ordering::Release);
        self.committed_addr.store(0, Ordering::Release);
        self.tail.store(0, Ordering::Release);
        drop(adv);
    }

    // --- watermark getters (for tests + later stages) ---
    #[inline]
    pub(crate) fn committed_addr(&self) -> u64 {
        self.committed_addr.load(Ordering::Acquire)
    }
    #[inline]
    pub(crate) fn committed_records(&self) -> u64 {
        self.committed_records.load(Ordering::Acquire)
    }
    #[inline]
    pub(crate) fn flush_cursor(&self) -> u64 {
        self.flush_cursor.load(Ordering::Acquire)
    }
    #[inline]
    pub(crate) fn tail(&self) -> u64 {
        self.tail.load(Ordering::Acquire)
    }

    /// Convenience for single-threaded callers/tests: reserve + fill +
    /// publish one record.
    pub(crate) fn append(&self, bytes: &[u8]) -> ReserveTicket {
        let t = self.reserve(bytes.len() as u64);
        self.fill(&t, bytes);
        self.publish(&t);
        t
    }
}

// ===========================================================================
// Standard (non-loom) unit tests
// ===========================================================================
#[cfg(all(test, not(loom)))]
mod tests {
    use super::*;
    use std::sync::Arc;

    fn rec(tag: u8, len: usize) -> Vec<u8> {
        vec![tag; len]
    }

    /// In-order single producer: dense advance, committed_addr == sum of
    /// lengths, and the flushed stream is byte-identical to the records.
    #[test]
    fn dense_advance_and_byte_identical() {
        let ring = WalRing::with_capacity(4096);
        let records = [rec(1, 10), rec(2, 20), rec(3, 5), rec(4, 33)];
        let mut expected = Vec::new();
        for (i, r) in records.iter().enumerate() {
            ring.append(r);
            assert_eq!(ring.committed_records(), (i + 1) as u64);
            expected.extend_from_slice(r);
        }
        assert_eq!(ring.committed_addr(), expected.len() as u64);

        let mut flushed = Vec::new();
        let copied = ring.copy_committed_prefix(&mut |s| flushed.extend_from_slice(s));
        assert_eq!(copied, expected.len() as u64);
        assert_eq!(flushed, expected, "flushed stream must equal record concat");
        // Second pass copies nothing (prefix already drained).
        assert_eq!(
            ring.copy_committed_prefix(&mut |_| panic!("nothing to copy")),
            0
        );
    }

    /// Out-of-order publish holds the prefix at the first un-published gap,
    /// then resumes when the gap is filled. File order == byte order.
    #[test]
    fn out_of_order_publish_holds_prefix() {
        let ring = WalRing::with_capacity(4096);
        let r1 = rec(0xA1, 8);
        let r2 = rec(0xB2, 16);
        let r3 = rec(0xC3, 4);
        // Reserve all three in order: byte ranges [0,8) [8,24) [24,28).
        let t1 = ring.reserve(r1.len() as u64);
        let t2 = ring.reserve(r2.len() as u64);
        let t3 = ring.reserve(r3.len() as u64);

        ring.fill(&t2, &r2);
        ring.publish(&t2);
        assert_eq!(ring.committed_addr(), 0, "byte 0 interval missing => stalls");
        assert_eq!(ring.committed_records(), 0);

        ring.fill(&t3, &r3);
        ring.publish(&t3);
        assert_eq!(ring.committed_addr(), 0, "still missing the [0,8) interval");

        ring.fill(&t1, &r1);
        ring.publish(&t1);
        assert_eq!(ring.committed_addr(), 28, "all published => prefix folds");
        assert_eq!(ring.committed_records(), 3);

        let mut flushed = Vec::new();
        ring.copy_committed_prefix(&mut |s| flushed.extend_from_slice(s));
        let mut expected = r1.clone();
        expected.extend_from_slice(&r2);
        expected.extend_from_slice(&r3);
        assert_eq!(flushed, expected, "file order == byte order (r1++r2++r3)");
    }

    /// A record that wraps the physical ring copies out identically to a
    /// linear (non-wrapping) layout.
    #[test]
    fn wrapped_copy_equals_linear() {
        let ring = WalRing::with_capacity(64); // pow2 = 64
        assert_eq!(ring.capacity(), 64);
        // Fill+drain 48 bytes to push the cursor near the end.
        let pre = rec(0x11, 48);
        ring.append(&pre);
        let mut sink = Vec::new();
        ring.copy_committed_prefix(&mut |s| sink.extend_from_slice(s));
        assert_eq!(sink, pre);
        // Next 32-byte record straddles offset 48..64 then wraps to 0..16.
        let wrapping = (0u8..32).collect::<Vec<u8>>();
        let t = ring.append(&wrapping);
        assert!(
            (t.start & ring.mask) + 32 > ring.capacity,
            "test must exercise the wrap: start_off={}",
            t.start & ring.mask
        );
        let mut flushed = Vec::new();
        ring.copy_committed_prefix(&mut |s| flushed.extend_from_slice(s));
        assert_eq!(flushed, wrapping, "wrapped record reassembles in order");
    }

    /// Concurrent producers + a single flusher: the drained stream contains
    /// every record exactly once (byte/tail order), none lost or torn.
    #[test]
    fn concurrent_producers_single_flusher() {
        use std::sync::atomic::{AtomicBool, Ordering as O};
        use std::thread;

        const PRODUCERS: usize = 4;
        const PER: usize = 2000;
        const REC_LEN: usize = 24;
        let ring = Arc::new(WalRing::with_capacity(1 << 20)); // 1 MiB, no backpressure
        let done = Arc::new(AtomicBool::new(false));

        let flusher = {
            let ring = Arc::clone(&ring);
            let done = Arc::clone(&done);
            thread::spawn(move || {
                let mut out: Vec<u8> = Vec::new();
                loop {
                    let n = ring.copy_committed_prefix(&mut |s| out.extend_from_slice(s));
                    if n == 0 {
                        if done.load(O::Acquire) && ring.committed_addr() == ring.flush_cursor() {
                            break;
                        }
                        std::hint::spin_loop();
                    }
                }
                out
            })
        };

        let mut handles = Vec::new();
        for p in 0..PRODUCERS {
            let ring = Arc::clone(&ring);
            handles.push(thread::spawn(move || {
                for i in 0..PER {
                    let tag = (p * PER + i) as u32;
                    let mut r = vec![0u8; REC_LEN];
                    r[..4].copy_from_slice(&tag.to_le_bytes());
                    ring.append(&r);
                }
            }));
        }
        for h in handles {
            h.join().unwrap();
        }
        done.store(true, O::Release);
        let out = flusher.join().unwrap();

        let total = PRODUCERS * PER;
        assert_eq!(
            out.len(),
            total * REC_LEN,
            "every record flushed exactly once"
        );
        assert_eq!(ring.committed_records(), total as u64);
        let mut seen = vec![false; total];
        for chunk in out.chunks_exact(REC_LEN) {
            let tag = u32::from_le_bytes(chunk[..4].try_into().unwrap()) as usize;
            assert!(!seen[tag], "duplicate record {tag}");
            seen[tag] = true;
        }
        assert!(seen.iter().all(|&b| b), "no record may be lost");
    }

    /// End-to-end integration with the REAL WAL stack: draining the ring's
    /// committed prefix into the real `WalWriter` produces a byte-for-byte
    /// identical file to the direct (legacy) append path, and that file
    /// replays back to the original inserts in order through the real
    /// reader. This is the stage-2 golden-file / byte-identical-replay gate;
    /// it proves `copy_committed_prefix` composes with `append_encoded`'s
    /// opaque-byte append (incl. records split across the physical wrap) and
    /// the on-disk codec + torn-tail reader — unchanged.
    #[test]
    fn ring_to_walwriter_byte_identical_and_replays() {
        use crate::journal::codec::encode_insert_record;
        use crate::journal::reader::replay;
        use crate::journal::wal_op::WalOp;
        use crate::journal::writer::WalWriter;

        let tree_id = 7u64;
        // Varied key/value sizes so records straddle the physical wrap.
        let inputs: Vec<(u64, Vec<u8>, Vec<u8>)> = (0..64u64)
            .map(|i| {
                let key = format!("bucket/{:02}/object-{i}", i % 4).into_bytes();
                let value = vec![(i & 0xff) as u8; (i as usize % 37) + 1];
                (i + 1, key, value) // seq is 1-indexed
            })
            .collect();
        let records: Vec<Vec<u8>> = inputs
            .iter()
            .map(|(seq, k, v)| {
                let mut buf = Vec::new();
                encode_insert_record(&mut buf, *seq, tree_id, k, v);
                buf
            })
            .collect();

        let dir = tempfile::tempdir().unwrap();

        // Path A — direct append (what the legacy worker does per record).
        let path_a = dir.path().join("a.wal");
        {
            let mut w = WalWriter::open_or_create(&path_a, tree_id).unwrap();
            for r in &records {
                w.append_encoded(r).unwrap();
            }
            w.flush().unwrap();
        }

        // Path B — through the ring, drained incrementally into a real
        // WalWriter. Tiny capacity forces physical wrap; the per-record
        // drain frees ring space so stage-1's no-backpressure ring never
        // overruns.
        let path_b = dir.path().join("b.wal");
        {
            let ring = WalRing::with_capacity(256);
            let mut w = WalWriter::open_or_create(&path_b, tree_id).unwrap();
            for r in &records {
                assert!(r.len() as u64 <= ring.capacity());
                ring.append(r);
                ring.copy_committed_prefix(&mut |s| w.append_encoded(s).unwrap());
            }
            w.flush().unwrap();
        }

        // 1. Byte-for-byte identical WAL file.
        assert_eq!(
            std::fs::read(&path_a).unwrap(),
            std::fs::read(&path_b).unwrap(),
            "ring-produced WAL must be byte-identical to the direct path"
        );

        // 2. Replays through the real reader to the original inserts, in order.
        let mut got: Vec<(u64, Vec<u8>, Vec<u8>)> = Vec::new();
        replay(&path_b, |op, seq, _| {
            if let WalOp::Insert {
                key,
                value,
                tree_id: tid,
                ..
            } = op
            {
                assert_eq!(*tid, tree_id);
                got.push((seq, key.clone(), value.clone()));
            }
            Ok(())
        })
        .unwrap();
        let expected: Vec<(u64, Vec<u8>, Vec<u8>)> = inputs
            .iter()
            .map(|(s, k, v)| (*s, k.clone(), v.clone()))
            .collect();
        assert_eq!(got, expected, "replay must round-trip records in order");
    }

    /// The durability-preserving drainer: a BACKGROUND flusher thread drains
    /// the committed prefix into the real `WalWriter` (whose 64KB auto-drain
    /// pushes async bytes to the OS promptly — preserving process-crash
    /// durability, unlike a drain-only-at-checkpoint shortcut) while N
    /// producers append concurrently to a small ring with manual
    /// backpressure (stage-5 preview). After teardown every record replays
    /// back exactly once through the real reader.
    #[test]
    fn background_flusher_concurrent_producers_replay() {
        use crate::journal::codec::encode_insert_record;
        use crate::journal::reader::replay;
        use crate::journal::wal_op::WalOp;
        use crate::journal::writer::WalWriter;
        use std::sync::atomic::{AtomicBool, Ordering as O};
        use std::sync::Mutex;
        use std::thread;

        const PRODUCERS: usize = 4;
        const PER: usize = 250;
        let tree_id = 9u64;
        // 4 KiB ring forces many wraps + reuse for 50KB+ of records; the
        // background flusher frees RAM and producers wait on
        // reserve_space_ready (stage-5 backpressure, previewed in the test).
        let ring = Arc::new(WalRing::with_capacity(4096));
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("bg.wal");
        let writer = Arc::new(Mutex::new(WalWriter::open_or_create(&path, tree_id).unwrap()));
        let stop = Arc::new(AtomicBool::new(false));

        let flusher = {
            let ring = Arc::clone(&ring);
            let writer = Arc::clone(&writer);
            let stop = Arc::clone(&stop);
            thread::spawn(move || loop {
                let n = ring.copy_committed_prefix(&mut |s| {
                    writer.lock().unwrap().append_encoded(s).unwrap();
                });
                if n == 0 {
                    if stop.load(O::Acquire) && ring.flush_cursor() == ring.committed_addr() {
                        break;
                    }
                    std::hint::spin_loop();
                }
            })
        };

        let mut handles = Vec::new();
        for p in 0..PRODUCERS {
            let ring = Arc::clone(&ring);
            handles.push(thread::spawn(move || {
                for i in 0..PER {
                    let seq = (p * PER + i + 1) as u64;
                    let key = format!("p{p}/obj-{i}").into_bytes();
                    let value = vec![(seq & 0xff) as u8; (i % 29) + 1];
                    let mut rec = Vec::new();
                    encode_insert_record(&mut rec, seq, tree_id, &key, &value);
                    assert!(rec.len() as u64 <= ring.capacity());
                    let t = ring.reserve(rec.len() as u64);
                    while !ring.reserve_space_ready(&t) {
                        std::hint::spin_loop();
                    }
                    ring.fill(&t, &rec);
                    ring.publish(&t);
                }
            }));
        }
        for h in handles {
            h.join().unwrap();
        }
        stop.store(true, O::Release);
        flusher.join().unwrap();
        writer.lock().unwrap().flush().unwrap();

        let total = PRODUCERS * PER;
        assert_eq!(ring.committed_records(), total as u64);

        let mut seqs = std::collections::BTreeSet::new();
        replay(&path, |op, seq, _| {
            if let WalOp::Insert { tree_id: tid, .. } = op {
                assert_eq!(*tid, tree_id);
                assert!(seqs.insert(seq), "duplicate seq {seq} on replay");
            }
            Ok(())
        })
        .unwrap();
        assert_eq!(seqs.len(), total, "every record must replay exactly once");
        assert_eq!(*seqs.iter().next().unwrap(), 1);
        assert_eq!(*seqs.iter().next_back().unwrap(), total as u64);
    }

    /// `reset_after_drain` (the truncate path) zeroes the byte cursors but
    /// keeps the record count, and appends resume from byte 0.
    #[test]
    fn reset_after_drain_keeps_record_count() {
        let ring = WalRing::with_capacity(256);
        for i in 0..3u8 {
            ring.append(&[i + 1; 20]);
        }
        let mut sink = Vec::new();
        ring.copy_committed_prefix(&mut |s| sink.extend_from_slice(s));
        assert_eq!(ring.committed_records(), 3);
        assert_eq!((ring.tail(), ring.committed_addr(), ring.flush_cursor()), (60, 60, 60));

        ring.reset_after_drain();
        assert_eq!((ring.tail(), ring.committed_addr(), ring.flush_cursor()), (0, 0, 0));
        assert_eq!(ring.committed_records(), 3, "record count survives truncate reset");

        let t = ring.append(&[9u8; 10]);
        assert_eq!(t.start, 0, "appends resume from byte 0 after reset");
        assert_eq!(ring.committed_records(), 4);
        let mut sink2 = Vec::new();
        ring.copy_committed_prefix(&mut |s| sink2.extend_from_slice(s));
        assert_eq!(sink2, vec![9u8; 10]);
    }
}

// ===========================================================================
// loom model: gap-safety + memory-ordering of reserve->fill->publish->copy.
// Run with:  RUSTFLAGS="--cfg loom" cargo test --lib journal::ring::loom
// ===========================================================================
#[cfg(loom)]
mod loom_tests {
    use super::*;
    use loom::sync::Arc;
    use loom::thread;

    /// Two concurrent producers + one flusher racing them. loom exhaustively
    /// explores interleavings and asserts the flusher NEVER observes a torn /
    /// gap copy: every byte below `committed_addr` was fully written by the
    /// publishing writer (payload byte == 1 or 2; a gap surfaces as 0).
    #[test]
    fn gap_safety_two_producers_one_flusher() {
        loom::model(|| {
            let ring = Arc::new(WalRing::with_capacity(64));

            let w1 = {
                let ring = Arc::clone(&ring);
                thread::spawn(move || ring.append(&[1u8; 4]))
            };
            let w2 = {
                let ring = Arc::clone(&ring);
                thread::spawn(move || ring.append(&[2u8; 4]))
            };

            let mut drained: Vec<u8> = Vec::new();
            ring.copy_committed_prefix(&mut |s| drained.extend_from_slice(s));
            for &b in &drained {
                assert!(b == 1 || b == 2, "torn/gap byte {b} observed by flusher");
            }

            w1.join().unwrap();
            w2.join().unwrap();

            let mut rest: Vec<u8> = Vec::new();
            ring.copy_committed_prefix(&mut |s| rest.extend_from_slice(s));
            for &b in &rest {
                assert!(b == 1 || b == 2, "torn/gap byte {b} in final drain");
            }
            assert_eq!(drained.len() + rest.len(), 8, "both records must drain");
            assert_eq!(ring.committed_records(), 2);
        });
    }

    /// Three concurrent publishers + a racing flusher. Stresses the `advance`
    /// lock (BTreeMap insert + contiguous fold) under more interleavings:
    /// asserts no torn/gap copy, dense final fold, and — implicitly — no
    /// deadlock (the lock is a leaf; the flusher never takes it).
    #[test]
    fn gap_safety_three_producers_one_flusher() {
        loom::model(|| {
            let ring = Arc::new(WalRing::with_capacity(64));
            let mut workers = Vec::new();
            for tag in 1u8..=3 {
                let ring = Arc::clone(&ring);
                workers.push(thread::spawn(move || ring.append(&[tag; 4])));
            }
            let mut drained: Vec<u8> = Vec::new();
            ring.copy_committed_prefix(&mut |s| drained.extend_from_slice(s));
            for &b in &drained {
                assert!((1..=3).contains(&b), "torn/gap byte {b}");
            }
            for w in workers {
                w.join().unwrap();
            }
            let mut rest: Vec<u8> = Vec::new();
            ring.copy_committed_prefix(&mut |s| rest.extend_from_slice(s));
            for &b in &rest {
                assert!((1..=3).contains(&b), "torn/gap byte {b} in final drain");
            }
            assert_eq!(drained.len() + rest.len(), 12, "all three must drain");
            assert_eq!(ring.committed_records(), 3);
        });
    }
}
