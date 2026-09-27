// Port of COLMAP's src/colmap/math/union_find_test.cc (1:1, same test names in snake_case).
// `UnionFind<std::string>` becomes `UnionFind<String>`; `Parents()` checks use
// `len`/`contains`/`parents()`.

use colmap_rust::math::union_find::UnionFind;
use std::collections::BTreeMap;

#[test]
fn union_find_default_constructor() {
    let mut uf = UnionFind::<i32>::new();
    // After construction, each element should be its own parent
    assert_eq!(uf.find(&1), 1);
    assert_eq!(uf.find(&2), 2);
    assert_eq!(uf.find(&3), 3);
}

#[test]
fn union_find_reserve() {
    // Test constructor with expected size parameter
    let mut uf = UnionFind::<i32>::new();
    uf.reserve(10);
    assert_eq!(uf.find(&1), 1);
    assert_eq!(uf.find(&2), 2);
}

#[test]
fn union_find_find_single_element() {
    let mut uf = UnionFind::<i32>::new();
    // Finding an element for the first time should return itself
    assert_eq!(uf.find(&42), 42);
    // Finding it again should still return itself
    assert_eq!(uf.find(&42), 42);
}

#[test]
fn union_find_union_two_elements() {
    let mut uf = UnionFind::<i32>::new();
    // Union two elements
    uf.union(&1, &2);
    // Both should have the same root
    assert_eq!(uf.find(&1), uf.find(&2));
}

#[test]
fn union_find_union_multiple_pairs() {
    let mut uf = UnionFind::<i32>::new();
    // Union multiple pairs
    uf.union(&1, &2);
    uf.union(&3, &4);
    uf.union(&5, &6);

    // Elements in the same set should have the same root
    assert_eq!(uf.find(&1), uf.find(&2));
    assert_eq!(uf.find(&3), uf.find(&4));
    assert_eq!(uf.find(&5), uf.find(&6));

    // Elements in different sets should have different roots
    assert_ne!(uf.find(&1), uf.find(&3));
    assert_ne!(uf.find(&1), uf.find(&5));
    assert_ne!(uf.find(&3), uf.find(&5));
}

#[test]
fn union_find_union_chain() {
    let mut uf = UnionFind::<i32>::new();
    // Create a chain: 1-2-3-4-5
    uf.union(&1, &2);
    uf.union(&2, &3);
    uf.union(&3, &4);
    uf.union(&4, &5);

    // All elements should have the same root
    let root = uf.find(&1);
    assert_eq!(uf.find(&2), root);
    assert_eq!(uf.find(&3), root);
    assert_eq!(uf.find(&4), root);
    assert_eq!(uf.find(&5), root);
}

#[test]
fn union_find_union_two_sets() {
    let mut uf = UnionFind::<i32>::new();
    // Create two separate sets
    uf.union(&1, &2);
    uf.union(&3, &4);

    // Verify they are separate
    assert_ne!(uf.find(&1), uf.find(&3));

    // Union the two sets
    uf.union(&2, &3);

    // Now all elements should have the same root
    let root = uf.find(&1);
    assert_eq!(uf.find(&2), root);
    assert_eq!(uf.find(&3), root);
    assert_eq!(uf.find(&4), root);
}

#[test]
fn union_find_union_same_element_twice() {
    let mut uf = UnionFind::<i32>::new();
    // Union an element with itself
    uf.union(&1, &1);
    assert_eq!(uf.find(&1), 1);

    // Union two elements, then union them again
    uf.union(&2, &3);
    let root = uf.find(&2);
    uf.union(&2, &3);
    // Should still have the same root
    assert_eq!(uf.find(&2), root);
    assert_eq!(uf.find(&3), root);
}

#[test]
fn union_find_path_compression() {
    let mut uf = UnionFind::<i32>::new();
    // Create a long chain: 1 -> 2 -> 3 -> 4 -> 5
    uf.union(&1, &2);
    uf.union(&2, &3);
    uf.union(&3, &4);
    uf.union(&4, &5);

    // Find root of element 1
    let root = uf.find(&1);

    // After path compression, finding 1 again should be efficient
    // The root should remain the same
    assert_eq!(uf.find(&1), root);

    // All elements should still have the same root
    assert_eq!(uf.find(&2), root);
    assert_eq!(uf.find(&3), root);
    assert_eq!(uf.find(&4), root);
    assert_eq!(uf.find(&5), root);
}

#[test]
fn union_find_compress() {
    let mut uf = UnionFind::<i32>::new();
    uf.union(&1, &2);
    uf.union(&2, &3);

    uf.compress();

    let parents: Vec<i32> = uf.parents().map(|(_, &p)| p).collect();
    for parent in parents {
        assert_eq!(parent, uf.find(&parent));
    }
}

#[test]
fn union_find_large_number_of_elements() {
    const K_NUM_ELEMENTS: i32 = 1000;
    let mut uf = UnionFind::<i32>::new();
    uf.reserve(K_NUM_ELEMENTS as usize);
    // Union many elements
    for i in 0..K_NUM_ELEMENTS {
        uf.union(&i, &((i + 1) % K_NUM_ELEMENTS));
    }

    // All elements should be in the same set
    let root = uf.find(&0);
    for i in 1..K_NUM_ELEMENTS {
        assert_eq!(uf.find(&i), root);
    }
}

#[test]
fn union_find_multiple_disjoint_sets() {
    let mut uf = UnionFind::<i32>::new();
    // Create multiple disjoint sets
    // Set 1: {1, 2, 3}
    uf.union(&1, &2);
    uf.union(&2, &3);

    // Set 2: {10, 20, 30}
    uf.union(&10, &20);
    uf.union(&20, &30);

    // Set 3: {100, 200, 300}
    uf.union(&100, &200);
    uf.union(&200, &300);

    // Verify elements in the same set have the same root
    assert_eq!(uf.find(&1), uf.find(&2));
    assert_eq!(uf.find(&2), uf.find(&3));
    assert_eq!(uf.find(&10), uf.find(&20));
    assert_eq!(uf.find(&20), uf.find(&30));
    assert_eq!(uf.find(&100), uf.find(&200));
    assert_eq!(uf.find(&200), uf.find(&300));

    // Verify elements in different sets have different roots
    assert_ne!(uf.find(&1), uf.find(&10));
    assert_ne!(uf.find(&1), uf.find(&100));
    assert_ne!(uf.find(&10), uf.find(&100));
}

#[test]
fn union_find_string_type() {
    let mut uf = UnionFind::<String>::new();
    let s = |x: &str| x.to_string();
    // Test with string type
    uf.union(&s("apple"), &s("apricot"));
    uf.union(&s("banana"), &s("blueberry"));
    uf.union(&s("apricot"), &s("avocado"));

    // Elements in the same set should have the same root
    assert_eq!(uf.find(&s("apple")), uf.find(&s("apricot")));
    assert_eq!(uf.find(&s("apple")), uf.find(&s("avocado")));
    assert_eq!(uf.find(&s("banana")), uf.find(&s("blueberry")));

    // Elements in different sets should have different roots
    assert_ne!(uf.find(&s("apple")), uf.find(&s("banana")));
}

#[test]
fn union_find_star_topology() {
    let mut uf = UnionFind::<i32>::new();
    // Create a star topology: center = 0, connected to 1, 2, 3, 4, 5
    for i in 1..=5 {
        uf.union(&0, &i);
    }

    // All elements should be in the same set
    for i in 1..=5 {
        assert_eq!(uf.find(&0), uf.find(&i));
    }
}

#[test]
fn union_find_reverse_union() {
    let mut uf = UnionFind::<i32>::new();
    // Union in reverse order
    uf.union(&5, &4);
    uf.union(&4, &3);
    uf.union(&3, &2);
    uf.union(&2, &1);

    // All should be in the same set
    let root = uf.find(&5);
    for i in 1..=5 {
        assert_eq!(uf.find(&i), root);
    }
}

#[test]
fn union_find_find_if_exists() {
    let mut uf = UnionFind::<i32>::new();
    let missing = uf.find_if_exists(&42);
    assert!(missing.is_none());

    assert_eq!(uf.find(&42), 42);
    let root = uf.find_if_exists(&42);
    assert!(root.is_some());
    assert_eq!(root.unwrap(), 42);

    assert!(uf.find_if_exists(&1).is_none());
    assert!(uf.find_if_exists(&2).is_none());
    uf.union(&2, &1);
    assert!(uf.find_if_exists(&1).is_some());
    assert!(uf.find_if_exists(&2).is_some());
}

#[test]
fn union_find_parents() {
    let mut uf = UnionFind::<i32>::new();
    assert!(uf.is_empty());

    uf.find(&1);
    assert_eq!(uf.len(), 1);
    assert!(uf.contains(&1));

    uf.union(&2, &3);
    assert_eq!(uf.len(), 3);
    assert!(uf.contains(&2));
    assert!(uf.contains(&3));

    uf.union(&1, &2);
    assert_eq!(uf.len(), 3);
}

#[test]
fn union_find_parents_group_by_root() {
    let mut uf = UnionFind::<i32>::new();
    uf.union(&1, &2);
    uf.union(&2, &3);
    uf.union(&10, &20);

    uf.compress();
    let mut groups: BTreeMap<i32, Vec<i32>> = BTreeMap::new();
    for (&elem, &parent) in uf.parents() {
        groups.entry(parent).or_default().push(elem);
    }

    assert_eq!(groups.len(), 2);
    assert_eq!(groups[&uf.find(&1)].len(), 3);
    assert_eq!(groups[&uf.find(&10)].len(), 2);
}

// Rust-only: `parents()` lists elements in insertion order (docs/CPP_DIVERGENCES.md,
// entry 42), and roots follow COLMAP's rule (root_x hangs under root_y).
#[test]
fn rust_only_union_find_parents_in_insertion_order() {
    let mut uf = UnionFind::<i32>::new();
    uf.union(&5, &3);
    uf.union(&9, &5);
    uf.find(&1);
    uf.compress();
    let parents: Vec<(i32, i32)> = uf.parents().map(|(&e, &p)| (e, p)).collect();
    assert_eq!(parents, vec![(5, 3), (3, 3), (9, 3), (1, 1)]);
}
