//! Port of COLMAP's `colmap/math/union_find.h`: a disjoint-set forest keyed on arbitrary
//! values (image ids, observations, strings) with path compression and no union by rank.
//! [`super::connected_components`] and [`super::spanning_tree`] build on it. Port of
//! colmap-sharp's `Mathematics/UnionFind.cs`. Tests: `tests/math/union_find.rs`
//! (union_find_test.cc 1:1).
//!
//! Tier A (exact): which element becomes a root is the same as COLMAP's for the same
//! sequence of calls (`union` always hangs root_x under root_y).
//!
//! Translation notes:
//! - COLMAP stores parents in a `NodeHashMap` (`boost::unordered_node_map`), whose iteration
//!   order is a function of Boost's hash layout. Here [`UnionFind::parents`] iterates in
//!   insertion order: entries live in a `Vec` and a `HashMap` only maps each element to its
//!   slot (it is never iterated). See docs/CPP_DIVERGENCES.md, entry 42.
//! - The C++ `Find` recurses; this one walks the path twice (find the root, then repoint
//!   every node on the path at it), which leaves the map in the same state without recursion
//!   depth limits on long chains.
//! - The optional `Hash` template parameter is dropped: keys use their `Hash` impl.

use std::collections::HashMap;
use std::hash::Hash;

/// Port of `colmap::UnionFind`: a disjoint-set structure over values of type `T`. Elements
/// are inserted lazily the first time they are seen.
#[derive(Clone, Debug)]
pub struct UnionFind<T> {
    // (element, parent) in insertion order; a root is its own parent.
    entries: Vec<(T, T)>,
    // Element -> index into `entries`. Lookup only, never iterated.
    index: HashMap<T, usize>,
}

impl<T: Clone + Eq + Hash> Default for UnionFind<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Clone + Eq + Hash> UnionFind<T> {
    /// Creates an empty structure.
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            index: HashMap::new(),
        }
    }

    /// `Reserve(capacity)`.
    pub fn reserve(&mut self, capacity: usize) {
        self.entries.reserve(capacity);
        self.index.reserve(capacity);
    }

    /// `Find(x)`: the root of `x`, compressing the path to it. If `x` is not in the structure,
    /// it is inserted as its own parent.
    pub fn find(&mut self, x: &T) -> T {
        let Some(&start) = self.index.get(x) else {
            self.index.insert(x.clone(), self.entries.len());
            self.entries.push((x.clone(), x.clone()));
            return x.clone();
        };
        // Walk to the root by slot index.
        let mut root = start;
        loop {
            let parent = self.index[&self.entries[root].1];
            if parent == root {
                break;
            }
            root = parent;
        }
        // Path compression: every node on the path now points straight at the root.
        let root_value = self.entries[root].0.clone();
        let mut node = start;
        while node != root {
            let next = self.index[&self.entries[node].1];
            self.entries[node].1 = root_value.clone();
            node = next;
        }
        root_value
    }

    /// `FindIfExists(x)`: the stored parent of `x` if it is in the structure (the root only
    /// after [`UnionFind::compress`] or a [`UnionFind::find`] of `x`), otherwise `None`.
    /// Inserts nothing.
    pub fn find_if_exists(&self, x: &T) -> Option<T> {
        self.index.get(x).map(|&i| self.entries[i].1.clone())
    }

    /// `Union(x, y)`: unites the sets containing `x` and `y`.
    pub fn union(&mut self, x: &T, y: &T) {
        let root_x = self.find(x);
        let root_y = self.find(y);
        if root_x != root_y {
            let i = self.index[&root_x];
            self.entries[i].1 = root_y;
        }
    }

    /// `Compress()`: path-compresses all elements so each points directly to its root. Call
    /// this once after all unions and before iterating [`UnionFind::parents`].
    pub fn compress(&mut self) {
        for i in 0..self.entries.len() {
            let elem = self.entries[i].0.clone();
            let root = self.find(&elem);
            self.entries[i].1 = root;
        }
    }

    /// `Parents()`: all elements and their parents, in insertion order.
    pub fn parents(&self) -> impl ExactSizeIterator<Item = (&T, &T)> + '_ {
        self.entries.iter().map(|(e, p)| (e, p))
    }

    /// `Parents().size()`.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// `Parents().empty()`.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// `Parents().count(x) != 0`.
    pub fn contains(&self, x: &T) -> bool {
        self.index.contains_key(x)
    }
}
