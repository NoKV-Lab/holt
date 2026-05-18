//! Concurrency primitives.
//!
//! `HybridLatch` is a 3-mode latch held per blob frame. The
//! contract follows LeanStore (Leis et al., ICDE 2018).

mod hybrid_latch;

pub use hybrid_latch::{Guard, GuardState, HybridLatch, LatchMode};
