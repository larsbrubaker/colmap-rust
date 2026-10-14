//! Port of the parts of PoseLib (BSD-3-Clause, Copyright (c) 2020 Viktor Larsson; see
//! THIRD_PARTY_NOTICES.md) that COLMAP's `estimators/solvers/` call, at commit
//! fa7280fee27f97aff31ae7f98bab7f583fac7d08 (COLMAP 4.2.0's pin), ported from colmap-sharp's
//! `Estimators/Solvers/PoseLib/`, one file per PoseLib source:
//!
//! - [`sturm`]: `misc/sturm.h`, Sturm-sequence root bracketing and Danilevsky's
//!   characteristic polynomial.

pub mod sturm;
