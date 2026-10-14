// Port of colmap/scene/track_test.cc, 1:1 (same test names, values and checks), for
// colmap_rust::scene::track::{Track, TrackElement}. `operator<<` becomes `Display`; the two
// `AddElement` / `DeleteElement` overloads taking ids are `add_element_ids` /
// `delete_element_ids`.

use colmap_rust::scene::track::{Track, TrackElement};
use colmap_rust::util::types::{INVALID_IMAGE_ID, INVALID_POINT2D_IDX};

#[test]
fn track_element_empty() {
    let track_el = TrackElement::default();
    assert_eq!(track_el.image_id, INVALID_IMAGE_ID);
    assert_eq!(track_el.point2d_idx, INVALID_POINT2D_IDX);
}

#[test]
fn track_element_equals() {
    let mut track_el = TrackElement::default();
    let mut other = track_el;
    assert_eq!(track_el, other);
    track_el.image_id = 1;
    assert_ne!(track_el, other);
    other.image_id = 1;
    assert_eq!(track_el, other);
}

#[test]
fn track_element_print() {
    let track_el = TrackElement::new(1, 2);
    assert_eq!(
        track_el.to_string(),
        "TrackElement(image_id=1, point2D_idx=2)"
    );
}

#[test]
fn track_default() {
    let track = Track::new();
    assert_eq!(track.length(), 0);
    assert_eq!(track.elements().len(), track.length());
}

#[test]
fn track_equals() {
    let mut track = Track::new();
    let mut other = track.clone();
    assert_eq!(track, other);
    track.add_element_ids(0, 1);
    assert_ne!(track, other);
    other.add_element_ids(0, 1);
    assert_eq!(track, other);
}

#[test]
fn track_print() {
    let mut track = Track::new();
    track.add_element_ids(1, 2);
    track.add_element_ids(2, 3);
    assert_eq!(
        track.to_string(),
        "Track(elements=[TrackElement(image_id=1, point2D_idx=2), \
         TrackElement(image_id=2, point2D_idx=3)])"
    );
}

#[test]
fn track_set_elements() {
    let mut track = Track::new();
    let elements = vec![TrackElement::new(0, 1), TrackElement::new(0, 2)];
    track.set_elements(elements);
    assert_eq!(track.length(), 2);
    assert_eq!(track.elements().len(), track.length());
    assert_eq!(track.element(0).image_id, 0);
    assert_eq!(track.element(0).point2d_idx, 1);
    assert_eq!(track.element(1).image_id, 0);
    assert_eq!(track.element(1).point2d_idx, 2);
    for i in 0..track.length() {
        assert_eq!(track.element(i).image_id, track.elements()[i].image_id);
        assert_eq!(
            track.element(i).point2d_idx,
            track.elements()[i].point2d_idx
        );
    }
}

#[test]
fn track_add_element() {
    let mut track = Track::new();
    track.add_element_ids(0, 1);
    track.add_element(TrackElement::new(0, 2));
    let elements = vec![TrackElement::new(0, 1), TrackElement::new(0, 2)];
    track.add_elements(&elements);
    assert_eq!(track.length(), 4);
    assert_eq!(track.elements().len(), track.length());
    assert_eq!(track.element(0).image_id, 0);
    assert_eq!(track.element(0).point2d_idx, 1);
    assert_eq!(track.element(1).image_id, 0);
    assert_eq!(track.element(1).point2d_idx, 2);
    assert_eq!(track.element(2).image_id, 0);
    assert_eq!(track.element(2).point2d_idx, 1);
    assert_eq!(track.element(3).image_id, 0);
    assert_eq!(track.element(3).point2d_idx, 2);
    for i in 0..track.length() {
        assert_eq!(track.element(i).image_id, track.elements()[i].image_id);
        assert_eq!(
            track.element(i).point2d_idx,
            track.elements()[i].point2d_idx
        );
    }
}

#[test]
fn track_delete_element() {
    let mut track = Track::new();
    track.add_element_ids(0, 1);
    track.add_element_ids(0, 2);
    track.add_element_ids(0, 3);
    track.add_element_ids(0, 3);
    assert_eq!(track.length(), 4);
    assert_eq!(track.elements().len(), track.length());
    track.delete_element(0).unwrap();
    assert_eq!(track.length(), 3);
    assert_eq!(track.elements().len(), track.length());
    assert_eq!(track.element(0).image_id, 0);
    assert_eq!(track.element(0).point2d_idx, 2);
    assert_eq!(track.element(1).image_id, 0);
    assert_eq!(track.element(1).point2d_idx, 3);
    assert_eq!(track.element(2).image_id, 0);
    assert_eq!(track.element(2).point2d_idx, 3);
    track.delete_element_ids(0, 3);
    assert_eq!(track.length(), 1);
    assert_eq!(track.elements().len(), track.length());
    assert_eq!(track.element(0).image_id, 0);
    assert_eq!(track.element(0).point2d_idx, 2);
}

#[test]
fn track_reserve() {
    let mut track = Track::new();
    track.reserve(2);
    assert_eq!(track.elements().capacity(), 2);
}

#[test]
fn track_compress() {
    let mut track = Track::new();
    track.add_element_ids(0, 1);
    track.add_element_ids(0, 2);
    track.add_element_ids(0, 3);
    track.add_element_ids(0, 3);
    assert_eq!(track.elements().capacity(), 4);
    track.delete_element(0).unwrap();
    track.delete_element(0).unwrap();
    assert_eq!(track.elements().capacity(), 4);
    track.compress();
    assert_eq!(track.elements().capacity(), 2);
}
