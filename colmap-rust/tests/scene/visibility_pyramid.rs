// Port of colmap/scene/visibility_pyramid_test.cc, 1:1 (same test names, values and
// checks), for colmap_rust::scene::visibility_pyramid::VisibilityPyramid. `Eigen::VectorXi`
// scores become a `Vec<usize>`; `scores.tail(n - 1).sum()` is the sum without the first.

use colmap_rust::scene::visibility_pyramid::VisibilityPyramid;

#[test]
fn visibility_pyramid_default() {
    let pyramid = VisibilityPyramid::default();
    assert_eq!(pyramid.num_levels(), 0);
    assert_eq!(pyramid.width(), 0);
    assert_eq!(pyramid.height(), 0);
    assert_eq!(pyramid.score(), 0);
}

#[test]
fn visibility_pyramid_score() {
    for num_levels in 1..8usize {
        let mut scores = vec![0usize; num_levels];
        let mut max_score = 0usize;
        for i in 1..=num_levels {
            scores[i - 1] = (1 << i) * (1 << i);
            max_score += scores[i - 1] * scores[i - 1];
        }
        let sum: usize = scores.iter().sum();
        let tail_sum: usize = scores[1..].iter().sum();

        let mut pyramid = VisibilityPyramid::new(num_levels, 4, 4);
        assert_eq!(pyramid.num_levels(), num_levels);
        assert_eq!(pyramid.width(), 4);
        assert_eq!(pyramid.height(), 4);
        assert_eq!(pyramid.score(), 0);
        assert_eq!(pyramid.max_score(), max_score);

        assert_eq!(pyramid.score(), 0);
        pyramid.set_point(0.0, 0.0).unwrap();
        assert_eq!(pyramid.score(), sum);
        pyramid.set_point(0.0, 0.0).unwrap();
        assert_eq!(pyramid.score(), sum);
        pyramid.set_point(0.0, 1.0).unwrap();
        assert_eq!(pyramid.score(), sum + tail_sum);
        pyramid.set_point(0.0, 1.0).unwrap();
        pyramid.set_point(0.0, 1.0).unwrap();
        pyramid.set_point(1.0, 0.0).unwrap();
        assert_eq!(pyramid.score(), sum + 2 * tail_sum);
        pyramid.set_point(1.0, 0.0).unwrap();
        pyramid.set_point(1.0, 1.0).unwrap();
        assert_eq!(pyramid.score(), sum + 3 * tail_sum);
        pyramid.reset_point(0.0, 0.0).unwrap();
        assert_eq!(pyramid.score(), sum + 3 * tail_sum);
        pyramid.reset_point(0.0, 0.0).unwrap();
        assert_eq!(pyramid.score(), sum + 2 * tail_sum);
        pyramid.set_point(0.0, 2.0).unwrap();
        assert_eq!(pyramid.score(), 2 * sum + 2 * tail_sum);
    }
}
