//! Approximate minimum degree (AMD) fill-reducing ordering that the sparse Cholesky
//! ([`SimplicialCholesky`](super::SimplicialCholesky)) applies before factoring, the
//! replacement for `Eigen::AMDOrdering` / CHOLMOD's AMD that COLMAP's sparse solvers use by
//! default. Port of colmap-sharp's `ColmapSharp/LinearAlgebra/AmdOrdering.cs` (MIT, our own
//! code).
//!
//! Written from the paper: P. R. Amestoy, T. A. Davis and I. S. Duff, "An Approximate
//! Minimum Degree Ordering Algorithm", SIAM J. Matrix Anal. Appl. 17(4), 1996. No
//! SuiteSparse/CSparse code was read or ported (CSparse is LGPL, docs/LICENSE_AUDIT.md), and
//! the data structures here are our own (per-node lists rather than the paper's single
//! in-place workspace). What is taken from the paper:
//! - The quotient graph: eliminated pivots become "elements" whose variable lists L_e stand
//!   for the cliques their elimination creates; each variable i keeps a list of adjacent
//!   elements E_i and of still-uncovered original neighbors A_i.
//! - The approximate external degree (the paper's equation for d_i): the minimum of
//!   (n - k - |i|), (d_i_old + |L_p \ i|) and (|A_i \ i| + |L_p \ i| + sum over e in E_i\p of
//!   |L_e \ L_p|), with |L_e \ L_p| computed for all touched elements in one pass over L_p.
//! - Element absorption (every element adjacent to the pivot is absorbed) and aggressive
//!   absorption (an element with |L_e \ L_p| = 0 is absorbed into the new element).
//! - Supervariable detection by hashing each variable's (E_i, A_i) and comparing lists of
//!   equal hash; indistinguishable variables are merged and ordered consecutively.
//!
//! Omitted as optimizations only (the result is still a valid, fill-reducing ordering):
//! dense-row deferral, initial supervariable detection and mass elimination. The permutation
//! differs from CHOLMOD's and Eigen's (docs/CPP_DIVERGENCES.md, entry 33).
//!
//! Every weight is a count of original variables, so |x| below means the summed weights.
//! Pivots are taken from the head of the lowest non-empty degree bucket; buckets are LIFO
//! lists, which makes the ordering deterministic for a given input.

use std::collections::HashMap;

use super::SparseMatrixCsc;
use crate::util::check::{ColmapError, ErrorKind};
use crate::Result;

/// "No node" in the bucket and member linked lists.
const NONE: usize = usize::MAX;

/// Fill-reducing ordering of the symmetric pattern A + A^T of a square matrix (diagonal
/// ignored). Returns `perm` with `perm[k]` = the original index eliminated k-th.
pub fn amd_ordering(a: &SparseMatrixCsc) -> Result<Vec<usize>> {
    amd_ordering_with_work(a).map(|(perm, _)| perm)
}

/// [`amd_ordering`], also returning the list entries the elimination visited: a
/// load-independent measure of its cost that tests bound to catch quadratic behavior (a
/// supervariable hash bucket or an element list that keeps growing).
pub fn amd_ordering_with_work(a: &SparseMatrixCsc) -> Result<(Vec<usize>, u64)> {
    if a.rows() != a.cols() {
        return Err(ColmapError::new(
            ErrorKind::InvalidArgument,
            format!("AMD needs a square matrix, got {}x{}.", a.rows(), a.cols()),
        ));
    }
    let mut adjacency = vec![Vec::new(); a.cols()];
    let (col_ptr, row_idx) = (a.col_ptr(), a.row_indices());
    for j in 0..a.cols() {
        for &i in &row_idx[col_ptr[j]..col_ptr[j + 1]] {
            if i != j {
                adjacency[i].push(j);
                adjacency[j].push(i);
            }
        }
    }
    Ok(order(&adjacency))
}

/// Fill-reducing ordering of an undirected graph given as adjacency lists (self loops and
/// repeated edges are tolerated; an edge listed in one direction only counts both ways).
/// Returns `perm` with `perm[k]` = the node eliminated k-th. A neighbor outside the graph is
/// an error.
pub fn amd_ordering_of_graph<L: AsRef<[usize]>>(adjacency: &[L]) -> Result<Vec<usize>> {
    let n = adjacency.len();
    for (i, list) in adjacency.iter().enumerate() {
        if let Some(&j) = list.as_ref().iter().find(|&&j| j >= n) {
            return Err(ColmapError::new(
                ErrorKind::OutOfRange,
                format!("Neighbor {j} of node {i} is outside 0..{}.", n as i64 - 1),
            ));
        }
    }
    Ok(order(adjacency).0)
}

fn order<L: AsRef<[usize]>>(adjacency: &[L]) -> (Vec<usize>, u64) {
    let n = adjacency.len();
    let mut lists: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (i, list) in adjacency.iter().enumerate() {
        for &j in list.as_ref() {
            if j != i {
                lists[i].push(j);
                lists[j].push(i);
            }
        }
    }
    // Sorted and deduplicated: colmap-sharp builds a HashSet per node and sorts it, so the
    // result does not depend on set enumeration order.
    for list in &mut lists {
        list.sort_unstable();
        list.dedup();
    }
    let mut state = State::new(lists);
    let perm = state.run();
    (perm, state.work)
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum NodeKind {
    Variable,
    NonPrincipal,
    Element,
    AbsorbedElement,
}

struct State {
    n: usize,
    kind: Vec<NodeKind>,
    weight: Vec<usize>,
    degree: Vec<usize>,
    var_adj: Vec<Vec<usize>>,
    elem_adj: Vec<Vec<usize>>,
    elem_vars: Vec<Vec<usize>>,
    elem_weight: Vec<usize>,
    // Degree buckets: doubly linked lists threaded through next/prev.
    head: Vec<usize>,
    next: Vec<usize>,
    prev: Vec<usize>,
    // Members of each supervariable, as a linked list from the principal variable.
    member_next: Vec<usize>,
    member_last: Vec<usize>,
    // Per-pivot scratch: variable marks, |L_e \ L_p| per element, list comparison marks.
    mark: Vec<u32>,
    w_stamp: Vec<u32>,
    w: Vec<i64>,
    cmp_mark: Vec<u32>,
    stamp: u32,
    cmp_stamp: u32,
    // List entries visited, for amd_ordering_with_work.
    work: u64,
}

impl State {
    fn new(var_adj: Vec<Vec<usize>>) -> Self {
        let n = var_adj.len();
        let mut state = Self {
            n,
            kind: vec![NodeKind::Variable; n],
            weight: vec![1; n],
            degree: var_adj.iter().map(Vec::len).collect(),
            var_adj,
            elem_adj: vec![Vec::new(); n],
            elem_vars: vec![Vec::new(); n],
            elem_weight: vec![0; n],
            head: vec![NONE; n + 1],
            next: vec![NONE; n],
            prev: vec![NONE; n],
            member_next: vec![NONE; n],
            member_last: (0..n).collect(),
            mark: vec![0; n],
            w_stamp: vec![0; n],
            w: vec![0; n],
            cmp_mark: vec![0; n],
            stamp: 0,
            cmp_stamp: 0,
            work: 0,
        };
        for i in 0..n {
            state.insert_bucket(i);
        }
        state
    }

    fn run(&mut self) -> Vec<usize> {
        let mut perm = Vec::with_capacity(self.n);
        let mut min_degree = 0;
        let mut lp = Vec::new();
        // Looked up only, never iterated, so its hash order cannot reach the output.
        let mut by_hash: HashMap<u64, Vec<usize>> = HashMap::new();
        while perm.len() < self.n {
            while self.head[min_degree] == NONE {
                self.work += 1;
                min_degree += 1;
            }
            let p = self.head[min_degree];
            self.remove_bucket(p);
            let mut m = p;
            while m != NONE {
                perm.push(m);
                m = self.member_next[m];
            }

            self.build_pivot_element(p, &mut lp);
            self.compute_external_element_weights(&lp);
            self.update_lists(p, &lp);
            self.merge_supervariables(&mut lp, &mut by_hash);

            let lp_weight: usize = lp.iter().map(|&i| self.weight[i]).sum();
            let remaining = self.n - perm.len();
            for &i in &lp {
                let degree = self.approximate_degree(i, p, lp_weight, remaining);
                self.degree[i] = degree;
                self.insert_bucket(i);
                min_degree = min_degree.min(degree);
            }
            self.elem_vars[p] = lp.clone();
            self.elem_weight[p] = lp_weight;
        }
        perm
    }

    /// Turns pivot p into an element: L_p is the union of A_p and of L_e over its adjacent
    /// elements, which are absorbed. Leaves the principal variables of L_p in `lp`, marked
    /// with the current stamp (p is marked too, so later passes skip it).
    fn build_pivot_element(&mut self, p: usize, lp: &mut Vec<usize>) {
        lp.clear();
        self.stamp += 1;
        self.mark[p] = self.stamp;
        self.work += (self.elem_adj[p].len() + self.var_adj[p].len()) as u64;
        let elems = std::mem::take(&mut self.elem_adj[p]);
        for &e in &elems {
            if self.kind[e] != NodeKind::Element {
                continue;
            }
            let vars = std::mem::take(&mut self.elem_vars[e]);
            self.work += vars.len() as u64;
            for &v in &vars {
                self.add_to_pivot_list(v, lp);
            }
            self.kind[e] = NodeKind::AbsorbedElement;
        }
        let vars = std::mem::take(&mut self.var_adj[p]);
        for &v in &vars {
            self.add_to_pivot_list(v, lp);
        }
        self.kind[p] = NodeKind::Element;
        for &i in lp.iter() {
            self.remove_bucket(i);
        }
    }

    fn add_to_pivot_list(&mut self, v: usize, lp: &mut Vec<usize>) {
        if self.kind[v] == NodeKind::Variable && self.mark[v] != self.stamp {
            self.mark[v] = self.stamp;
            lp.push(v);
        }
    }

    /// w(e) = |L_e \ L_p| for every live element adjacent to a variable of L_p.
    fn compute_external_element_weights(&mut self, lp: &[usize]) {
        for &i in lp {
            self.work += self.elem_adj[i].len() as u64;
            for k in 0..self.elem_adj[i].len() {
                let e = self.elem_adj[i][k];
                if self.kind[e] != NodeKind::Element {
                    continue;
                }
                if self.w_stamp[e] != self.stamp {
                    self.w_stamp[e] = self.stamp;
                    self.w[e] = self.elem_weight[e] as i64;
                }
                self.w[e] -= self.weight[i] as i64;
            }
        }
    }

    /// For each i in L_p: drop absorbed elements from E_i (aggressively absorbing any element
    /// now inside L_p) and add p; drop from A_i every variable covered by p.
    fn update_lists(&mut self, p: usize, lp: &[usize]) {
        for &i in lp {
            let mut elems = std::mem::take(&mut self.elem_adj[i]);
            self.work += (elems.len() + self.var_adj[i].len()) as u64;
            elems.retain(|&e| {
                if self.kind[e] != NodeKind::Element {
                    return false;
                }
                if self.w[e] == 0 {
                    self.kind[e] = NodeKind::AbsorbedElement;
                    self.elem_vars[e] = Vec::new();
                    return false;
                }
                true
            });
            elems.push(p);
            self.elem_adj[i] = elems;

            let (kind, mark, stamp) = (&self.kind, &self.mark, self.stamp);
            self.var_adj[i].retain(|&v| kind[v] == NodeKind::Variable && mark[v] != stamp);
        }
    }

    /// Merges variables of L_p with identical (E_i, A_i) into one supervariable, keeping the
    /// earliest in L_p as principal. Removes the merged ones from `lp`.
    fn merge_supervariables(
        &mut self,
        lp: &mut Vec<usize>,
        by_hash: &mut HashMap<u64, Vec<usize>>,
    ) {
        by_hash.clear();
        let mut kept = 0;
        for t in 0..lp.len() {
            let i = lp[t];
            self.work += (self.elem_adj[i].len() + self.var_adj[i].len()) as u64;
            let mut hash = 0u64;
            for &e in &self.elem_adj[i] {
                hash = hash.wrapping_add(e as u64);
            }
            for &v in &self.var_adj[i] {
                hash = hash.wrapping_add(v as u64);
            }

            let mut merged = false;
            let candidates = by_hash.entry(hash).or_default();
            for &j in candidates.iter() {
                self.work += 1;
                if self.same_lists(i, j) {
                    self.weight[j] += self.weight[i];
                    self.weight[i] = 0;
                    self.kind[i] = NodeKind::NonPrincipal;
                    self.member_next[self.member_last[j]] = i;
                    self.member_last[j] = self.member_last[i];
                    self.elem_adj[i] = Vec::new();
                    self.var_adj[i] = Vec::new();
                    merged = true;
                    break;
                }
            }
            if !merged {
                candidates.push(i);
                lp[kept] = i;
                kept += 1;
            }
        }
        lp.truncate(kept);
    }

    fn same_lists(&mut self, i: usize, j: usize) -> bool {
        let (ei, ej, ai, aj) = (
            &self.elem_adj[i],
            &self.elem_adj[j],
            &self.var_adj[i],
            &self.var_adj[j],
        );
        if ei.len() != ej.len() || ai.len() != aj.len() {
            return false;
        }
        self.work += (ei.len() + ai.len() + ej.len() + aj.len()) as u64;

        // Element ids and variable ids share one index space, and a node is never both, so
        // one mark pass over both lists of j compares the two sets at once.
        self.cmp_stamp += 1;
        let stamp = self.cmp_stamp;
        for &x in ej.iter().chain(aj) {
            self.cmp_mark[x] = stamp;
        }
        ei.iter().chain(ai).all(|&x| self.cmp_mark[x] == stamp)
    }

    /// The paper's approximate external degree of variable i after eliminating p.
    fn approximate_degree(
        &mut self,
        i: usize,
        p: usize,
        lp_weight: usize,
        remaining: usize,
    ) -> usize {
        self.work += (self.var_adj[i].len() + self.elem_adj[i].len()) as u64;
        let outside_i = lp_weight as i64 - self.weight[i] as i64;
        let mut bound3 = outside_i;
        for &v in &self.var_adj[i] {
            bound3 += self.weight[v] as i64;
        }
        for &e in &self.elem_adj[i] {
            if e != p {
                bound3 += self.w[e];
            }
        }
        let bound2 = self.degree[i] as i64 + outside_i;
        let bound1 = remaining as i64 - self.weight[i] as i64;
        bound1.min(bound2.min(bound3)).max(0) as usize
    }

    fn insert_bucket(&mut self, i: usize) {
        let d = self.degree[i];
        self.prev[i] = NONE;
        self.next[i] = self.head[d];
        if self.head[d] != NONE {
            self.prev[self.head[d]] = i;
        }
        self.head[d] = i;
    }

    fn remove_bucket(&mut self, i: usize) {
        if self.prev[i] != NONE {
            self.next[self.prev[i]] = self.next[i];
        } else {
            self.head[self.degree[i]] = self.next[i];
        }
        if self.next[i] != NONE {
            self.prev[self.next[i]] = self.prev[i];
        }
    }
}
