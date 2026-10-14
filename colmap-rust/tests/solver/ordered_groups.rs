// Port of Ceres Solver 2.2.0's internal/ceres/ordered_groups_test.cc (BSD-3-Clause, see
// THIRD_PARTY_NOTICES.md): `colmap_rust::solver::ParameterBlockOrdering`, same names and values.
// Ceres' elements are `double*`s into one array (x, x + 1, x + 2); here they are the
// parameter block ids 0, 1, 2. MinNonZeroGroup's death test becomes "returns Err".

use colmap_rust::solver::{ParameterBlockId, ParameterBlockOrdering};

const X: [ParameterBlockId; 3] = [
    ParameterBlockId(0),
    ParameterBlockId(1),
    ParameterBlockId(2),
];

#[test]
fn ordered_groups_empty_ordered_group_behaves_correctly() {
    let mut ordering = ParameterBlockOrdering::new();
    assert_eq!(ordering.num_groups(), 0);
    assert_eq!(ordering.num_elements(), 0);
    assert_eq!(ordering.group_size(1), 0);
    let x = ParameterBlockId(0);
    assert_eq!(ordering.group_id(x), -1);
    assert!(!ordering.remove(x));
}

#[test]
fn ordered_groups_everything_in_one_group() {
    let mut ordering = ParameterBlockOrdering::new();
    ordering.add_element_to_group(X[0], 1);
    ordering.add_element_to_group(X[1], 1);
    ordering.add_element_to_group(X[2], 1);
    ordering.add_element_to_group(X[0], 1);

    assert_eq!(ordering.num_groups(), 1);
    assert_eq!(ordering.num_elements(), 3);
    assert_eq!(ordering.group_size(1), 3);
    assert_eq!(ordering.group_size(0), 0);
    assert_eq!(ordering.group_id(X[0]), 1);
    assert_eq!(ordering.group_id(X[1]), 1);
    assert_eq!(ordering.group_id(X[2]), 1);

    ordering.remove(X[0]);
    assert_eq!(ordering.num_groups(), 1);
    assert_eq!(ordering.num_elements(), 2);
    assert_eq!(ordering.group_size(1), 2);
    assert_eq!(ordering.group_size(0), 0);

    assert_eq!(ordering.group_id(X[0]), -1);
    assert_eq!(ordering.group_id(X[1]), 1);
    assert_eq!(ordering.group_id(X[2]), 1);
}

#[test]
fn ordered_groups_start_in_one_group_and_then_split() {
    let mut ordering = ParameterBlockOrdering::new();
    ordering.add_element_to_group(X[0], 1);
    ordering.add_element_to_group(X[1], 1);
    ordering.add_element_to_group(X[2], 1);
    ordering.add_element_to_group(X[0], 1);

    assert_eq!(ordering.num_groups(), 1);
    assert_eq!(ordering.num_elements(), 3);
    assert_eq!(ordering.group_size(1), 3);
    assert_eq!(ordering.group_size(0), 0);
    assert_eq!(ordering.group_id(X[0]), 1);
    assert_eq!(ordering.group_id(X[1]), 1);
    assert_eq!(ordering.group_id(X[2]), 1);

    ordering.add_element_to_group(X[0], 5);
    assert_eq!(ordering.num_groups(), 2);
    assert_eq!(ordering.num_elements(), 3);
    assert_eq!(ordering.group_size(1), 2);
    assert_eq!(ordering.group_size(5), 1);
    assert_eq!(ordering.group_size(0), 0);

    assert_eq!(ordering.group_id(X[0]), 5);
    assert_eq!(ordering.group_id(X[1]), 1);
    assert_eq!(ordering.group_id(X[2]), 1);
}

#[test]
fn ordered_groups_add_and_remove_every_thing_from_one_group() {
    let mut ordering = ParameterBlockOrdering::new();
    ordering.add_element_to_group(X[0], 1);
    ordering.add_element_to_group(X[1], 1);
    ordering.add_element_to_group(X[2], 1);
    ordering.add_element_to_group(X[0], 1);

    assert_eq!(ordering.num_groups(), 1);
    assert_eq!(ordering.num_elements(), 3);
    assert_eq!(ordering.group_size(1), 3);
    assert_eq!(ordering.group_size(0), 0);
    assert_eq!(ordering.group_id(X[0]), 1);
    assert_eq!(ordering.group_id(X[1]), 1);
    assert_eq!(ordering.group_id(X[2]), 1);

    ordering.add_element_to_group(X[0], 5);
    ordering.add_element_to_group(X[1], 5);
    ordering.add_element_to_group(X[2], 5);
    assert_eq!(ordering.num_groups(), 1);
    assert_eq!(ordering.num_elements(), 3);
    assert_eq!(ordering.group_size(1), 0);
    assert_eq!(ordering.group_size(5), 3);
    assert_eq!(ordering.group_size(0), 0);

    assert_eq!(ordering.group_id(X[0]), 5);
    assert_eq!(ordering.group_id(X[1]), 5);
    assert_eq!(ordering.group_id(X[2]), 5);
}

#[test]
fn ordered_groups_reverse_ordering() {
    let mut ordering = ParameterBlockOrdering::new();
    ordering.add_element_to_group(X[0], 1);
    ordering.add_element_to_group(X[1], 2);
    ordering.add_element_to_group(X[2], 2);

    assert_eq!(ordering.num_groups(), 2);
    assert_eq!(ordering.num_elements(), 3);
    assert_eq!(ordering.group_size(1), 1);
    assert_eq!(ordering.group_size(2), 2);
    assert_eq!(ordering.group_id(X[0]), 1);
    assert_eq!(ordering.group_id(X[1]), 2);
    assert_eq!(ordering.group_id(X[2]), 2);

    ordering.reverse();

    assert_eq!(ordering.num_groups(), 2);
    assert_eq!(ordering.num_elements(), 3);
    assert_eq!(ordering.group_size(3), 1);
    assert_eq!(ordering.group_size(2), 2);
    assert_eq!(ordering.group_id(X[0]), 3);
    assert_eq!(ordering.group_id(X[1]), 2);
    assert_eq!(ordering.group_id(X[2]), 2);
}

#[test]
fn ordered_groups_reverse_ordering_with_empty_ordered_groups() {
    let mut ordering = ParameterBlockOrdering::new();

    // This should be a no-op.
    ordering.reverse();

    // Ensure the properties of an empty OrderedGroups still hold after Reverse().
    assert_eq!(ordering.num_groups(), 0);
    assert_eq!(ordering.num_elements(), 0);
    assert_eq!(ordering.group_size(1), 0);
    let x = ParameterBlockId(0);
    assert_eq!(ordering.group_id(x), -1);
    assert!(!ordering.remove(x));
}

#[test]
fn ordered_groups_bulk_remove() {
    let mut ordering = ParameterBlockOrdering::new();
    ordering.add_element_to_group(X[0], 1);
    ordering.add_element_to_group(X[1], 2);
    ordering.add_element_to_group(X[2], 2);

    let elements_to_remove = [X[0], X[2]];

    assert_eq!(ordering.remove_all(&elements_to_remove), 2);
    assert_eq!(ordering.num_elements(), 1);
    assert_eq!(ordering.group_id(X[0]), -1);
    assert_eq!(ordering.group_id(X[1]), 2);
    assert_eq!(ordering.group_id(X[2]), -1);
}

#[test]
fn ordered_groups_bulk_remove_with_no_elements() {
    let mut ordering = ParameterBlockOrdering::new();

    let mut elements_to_remove = vec![X[0], X[2]];

    assert_eq!(ordering.remove_all(&elements_to_remove), 0);

    ordering.add_element_to_group(X[0], 1);
    ordering.add_element_to_group(X[1], 2);
    ordering.add_element_to_group(X[2], 2);

    elements_to_remove.clear();
    assert_eq!(ordering.remove_all(&elements_to_remove), 0);
}

#[test]
fn ordered_groups_min_non_zero_group() {
    let mut ordering = ParameterBlockOrdering::new();

    ordering.add_element_to_group(X[0], 1);
    ordering.add_element_to_group(X[1], 1);
    ordering.add_element_to_group(X[2], 2);

    assert_eq!(ordering.min_non_zero_group().unwrap(), 1);
    ordering.remove(X[0]);

    assert_eq!(ordering.min_non_zero_group().unwrap(), 1);
    ordering.remove(X[1]);

    assert_eq!(ordering.min_non_zero_group().unwrap(), 2);
    ordering.remove(X[2]);

    // No non-zero groups left.
    assert!(ordering.min_non_zero_group().is_err());
}
