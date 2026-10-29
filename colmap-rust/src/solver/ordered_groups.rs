//! Elements in ordered integer groups, and [`ParameterBlockOrdering`], the user's ordering of
//! parameter blocks for the linear solver (lowest group eliminated first by the Schur
//! solvers). Port of Ceres Solver 2.2.0 `include/ceres/ordered_groups.h` (BSD-3-Clause, see
//! THIRD_PARTY_NOTICES.md); colmap-sharp's `ColmapSharp/Solver/ParameterBlockOrdering.cs` is
//! the C# form. COLMAP's `GlobalPositioner` is the one caller: scales in group 0, points in
//! group 1, frame centers and rig cameras in group 2.
//!
//! Ceres keeps each group in a `std::set<double*>`, so a group's blocks come out in heap-address
//! order, and the Schur solvers lay out the eliminated blocks (and the reduced camera system's
//! columns) in that order. Here the elements are [`crate::solver::ParameterBlockId`]s, so a
//! group's blocks come out in id order: the order they were added to the problem. That is a
//! deliberate divergence. Groups are still visited in increasing id order, and every other
//! rule (moving an element, `Reverse`'s numbering, the counts) is Ceres'. Address order is an
//! accident of the allocator, so no run-to-run order exists to match; the ids make Ceres' own
//! `std::set` rule deterministic. The effect on results is rounding-level (the order of
//! additions into the reduced system), within the Tier C outcome bar. colmap-sharp keeps each
//! group in the order its blocks were added to the group (its entry 36); id order is the same
//! wherever a caller adds blocks to the problem and to the ordering in the same order, as
//! COLMAP's `GlobalPositioner` does. Evidence: `tests/solver/ordered_groups.rs` (Ceres'
//! `ordered_groups_test.cc`, unchanged values) passes; none of its cases depends on the order
//! inside a group.

use std::collections::{BTreeMap, BTreeSet};

use super::problem::ParameterBlockId;
use crate::{check_ne, Result};

/// Port of `ceres::OrderedGroups<T>`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct OrderedGroups<T: Ord + Copy> {
    group_to_elements: BTreeMap<i32, BTreeSet<T>>,
    // Ceres uses an unordered_map; it is only looked up, never iterated where order shows.
    element_to_group: BTreeMap<T, i32>,
}

/// Port of `ceres::ParameterBlockOrdering` (`OrderedGroups<double*>`).
pub type ParameterBlockOrdering = OrderedGroups<ParameterBlockId>;

impl<T: Ord + Copy> OrderedGroups<T> {
    /// An empty ordering.
    pub fn new() -> Self {
        Self {
            group_to_elements: BTreeMap::new(),
            element_to_group: BTreeMap::new(),
        }
    }

    /// Adds `element` to `group`, moving it out of its current group. False (and nothing
    /// changes) for a negative group.
    pub fn add_element_to_group(&mut self, element: T, group: i32) -> bool {
        if group < 0 {
            return false;
        }

        if let Some(&current) = self.element_to_group.get(&element) {
            if current == group {
                return true;
            }
            self.remove_from_group(element, current);
        }

        self.element_to_group.insert(element, group);
        self.group_to_elements
            .entry(group)
            .or_default()
            .insert(element);
        true
    }

    /// Removes every element and group.
    pub fn clear(&mut self) {
        self.group_to_elements.clear();
        self.element_to_group.clear();
    }

    /// Removes `element`; false if it was not a member.
    pub fn remove(&mut self, element: T) -> bool {
        let current_group = self.group_id(element);
        if current_group < 0 {
            return false;
        }
        self.remove_from_group(element, current_group);
        self.element_to_group.remove(&element);
        true
    }

    /// Removes each of `elements`; returns how many were members.
    pub fn remove_all(&mut self, elements: &[T]) -> usize {
        if self.num_elements() == 0 || elements.is_empty() {
            return 0;
        }
        let mut num_removed = 0;
        for &element in elements {
            num_removed += usize::from(self.remove(element));
        }
        num_removed
    }

    /// Reverses the order of the groups: the highest keeps its id and the others follow with
    /// consecutive ids above it.
    pub fn reverse(&mut self) {
        let Some((&last, _)) = self.group_to_elements.last_key_value() else {
            return;
        };

        let old = std::mem::take(&mut self.group_to_elements);
        let mut groups = old.into_iter().rev();
        let mut new_group_to_elements = BTreeMap::new();
        if let Some((group, elements)) = groups.next() {
            new_group_to_elements.insert(group, elements);
        }

        for (new_group_id, (_, elements)) in (last + 1..).zip(groups) {
            for &element in &elements {
                self.element_to_group.insert(element, new_group_id);
            }
            new_group_to_elements.insert(new_group_id, elements);
        }
        self.group_to_elements = new_group_to_elements;
    }

    /// The element's group, or -1 if it is not a member.
    pub fn group_id(&self, element: T) -> i32 {
        self.element_to_group.get(&element).copied().unwrap_or(-1)
    }

    /// True if `element` is in some group.
    pub fn is_member(&self, element: T) -> bool {
        self.element_to_group.contains_key(&element)
    }

    /// Number of elements in `group` (0 if there is no such group).
    pub fn group_size(&self, group: i32) -> usize {
        self.group_to_elements.get(&group).map_or(0, BTreeSet::len)
    }

    /// Number of elements over all groups.
    pub fn num_elements(&self) -> usize {
        self.element_to_group.len()
    }

    /// Number of non-empty groups.
    pub fn num_groups(&self) -> usize {
        self.group_to_elements.len()
    }

    /// The lowest group id. Ceres CHECK-fails on an empty ordering; the port returns the
    /// check's error.
    pub fn min_non_zero_group(&self) -> Result<i32> {
        check_ne!(self.num_groups(), 0);
        Ok(self
            .group_to_elements
            .first_key_value()
            .map_or(0, |(&group, _)| group))
    }

    /// The groups in increasing id order, each with its elements in increasing order.
    pub fn group_to_elements(&self) -> &BTreeMap<i32, BTreeSet<T>> {
        &self.group_to_elements
    }

    /// Each element's group.
    pub fn element_to_group(&self) -> &BTreeMap<T, i32> {
        &self.element_to_group
    }

    fn remove_from_group(&mut self, element: T, group: i32) {
        if let Some(elements) = self.group_to_elements.get_mut(&group) {
            elements.remove(&element);
            if elements.is_empty() {
                self.group_to_elements.remove(&group);
            }
        }
    }
}
