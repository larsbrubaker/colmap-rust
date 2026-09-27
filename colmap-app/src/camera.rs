// Orbit camera for the 3D viewport: yaw / pitch / distance around a target point, turned into
// view and projection matrices. Pure math with no GUI or GPU, so it is unit-tested here and the
// viewport widget (`viewport.rs`) only maps pointer gestures onto `orbit` / `pan` / `zoom`.
//
// Not a COLMAP port: COLMAP's Qt viewer (`ui/model_viewer_widget.cc`) is excluded with Qt, and
// this is the usual turntable camera. Uses glam (MIT/Apache-2.0) because colmap-rust's own
// linalg does not exist yet and the core library must not depend on GUI math anyway.
//
// World convention: right-handed, **Z up** (the ground grid lies in the XY plane). Projection
// maps depth to wgpu's [0, 1] range (`Mat4::perspective_rh`).

use glam::{Mat4, Vec3};

/// Radians of yaw / pitch per logical pixel of orbit drag.
pub const ORBIT_RADIANS_PER_PIXEL: f32 = 0.01;
/// Pitch is kept strictly inside (-90°, 90°) so the look-at up vector never degenerates.
pub const MAX_PITCH: f32 = 89.0 * std::f32::consts::PI / 180.0;
/// Closest the eye may get to the target (world units).
pub const MIN_DISTANCE: f32 = 0.05;
/// Farthest the eye may get from the target (world units).
pub const MAX_DISTANCE: f32 = 10_000.0;
/// Wheel zoom: the distance is multiplied by `exp(-delta_y * ZOOM_PER_WHEEL_PIXEL)`, so a
/// forward wheel step (positive `delta_y` in agg-gui's convention) moves the eye closer.
pub const ZOOM_PER_WHEEL_PIXEL: f32 = 0.002;

/// Turntable camera orbiting `target`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OrbitCamera {
    /// Rotation about world Z, radians. 0 looks from +X toward the target.
    pub yaw: f32,
    /// Elevation above the XY plane, radians, clamped to ±[`MAX_PITCH`].
    pub pitch: f32,
    /// Eye-to-target distance, clamped to [[`MIN_DISTANCE`], [`MAX_DISTANCE`]].
    pub distance: f32,
    /// Point the camera orbits around and looks at.
    pub target: Vec3,
    /// Vertical field of view, radians.
    pub fov_y: f32,
}

impl Default for OrbitCamera {
    /// A three-quarter view of the origin that shows all three axes and the grid.
    fn default() -> Self {
        Self {
            yaw: 45.0_f32.to_radians(),
            pitch: 30.0_f32.to_radians(),
            distance: 12.0,
            target: Vec3::ZERO,
            fov_y: 45.0_f32.to_radians(),
        }
    }
}

impl OrbitCamera {
    /// Rotate by a pointer drag of `(dx, dy)` logical pixels (agg-gui Y-up, so `dy > 0` is an
    /// upward drag). Dragging right spins the scene right; dragging up tilts it toward the viewer.
    pub fn orbit(&mut self, dx: f32, dy: f32) {
        self.yaw -= dx * ORBIT_RADIANS_PER_PIXEL;
        self.yaw = self.yaw.rem_euclid(std::f32::consts::TAU);
        self.pitch = (self.pitch - dy * ORBIT_RADIANS_PER_PIXEL).clamp(-MAX_PITCH, MAX_PITCH);
    }

    /// Translate the target so the point under the cursor follows a drag of `(dx, dy)` logical
    /// pixels in a viewport `viewport_height` logical pixels tall.
    pub fn pan(&mut self, dx: f32, dy: f32, viewport_height: f32) {
        if viewport_height <= 0.0 {
            return;
        }
        let world_per_pixel = 2.0 * self.distance * (self.fov_y * 0.5).tan() / viewport_height;
        let (right, up) = self.screen_axes();
        self.target -= (right * dx + up * dy) * world_per_pixel;
    }

    /// Dolly toward (positive `delta_y`) or away from the target, clamped to the distance range.
    pub fn zoom(&mut self, delta_y: f32) {
        let factor = (-delta_y * ZOOM_PER_WHEEL_PIXEL).exp();
        self.distance = (self.distance * factor).clamp(MIN_DISTANCE, MAX_DISTANCE);
    }

    /// Unit vector from the target toward the eye.
    pub fn back_direction(&self) -> Vec3 {
        let (sy, cy) = self.yaw.sin_cos();
        let (sp, cp) = self.pitch.sin_cos();
        Vec3::new(cp * cy, cp * sy, sp)
    }

    /// World-space eye position.
    pub fn eye(&self) -> Vec3 {
        self.target + self.back_direction() * self.distance
    }

    /// Camera right and up vectors in world space (both unit length).
    pub fn screen_axes(&self) -> (Vec3, Vec3) {
        let forward = -self.back_direction();
        let right = forward.cross(Vec3::Z).normalize_or_zero();
        let up = right.cross(forward).normalize_or_zero();
        (right, up)
    }

    /// World → view matrix.
    pub fn view(&self) -> Mat4 {
        Mat4::look_at_rh(self.eye(), self.target, Vec3::Z)
    }

    /// View → clip matrix for a viewport of the given width / height ratio. Near and far scale
    /// with the orbit distance so depth precision follows the zoom level.
    pub fn projection(&self, aspect: f32) -> Mat4 {
        let aspect = if aspect.is_finite() && aspect > 0.0 {
            aspect
        } else {
            1.0
        };
        let near = (self.distance * 0.01).max(1e-4);
        let far = self.distance * 100.0 + 100.0;
        Mat4::perspective_rh(self.fov_y, aspect, near, far)
    }

    /// Projection × view.
    pub fn view_projection(&self, aspect: f32) -> Mat4 {
        self.projection(aspect) * self.view()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec4;

    fn approx(a: Vec3, b: Vec3) -> bool {
        (a - b).length() < 1e-4
    }

    #[test]
    fn eye_sits_at_distance_from_target() {
        let cam = OrbitCamera {
            target: Vec3::new(1.0, 2.0, 3.0),
            ..OrbitCamera::default()
        };
        assert!(((cam.eye() - cam.target).length() - cam.distance).abs() < 1e-4);
    }

    #[test]
    fn zero_yaw_and_pitch_looks_from_plus_x() {
        let cam = OrbitCamera {
            yaw: 0.0,
            pitch: 0.0,
            distance: 5.0,
            ..OrbitCamera::default()
        };
        assert!(approx(cam.eye(), Vec3::new(5.0, 0.0, 0.0)));
        let (right, up) = cam.screen_axes();
        assert!(approx(up, Vec3::Z));
        // Looking down -X with Z up: right = forward × Z = (-X) × Z = +Y.
        assert!(approx(right, Vec3::Y));
    }

    #[test]
    fn target_projects_to_clip_center() {
        let cam = OrbitCamera::default();
        let clip = cam.view_projection(16.0 / 9.0) * Vec4::new(0.0, 0.0, 0.0, 1.0);
        let ndc = clip / clip.w;
        assert!(ndc.x.abs() < 1e-5 && ndc.y.abs() < 1e-5);
        assert!(ndc.z > 0.0 && ndc.z < 1.0, "depth in wgpu range: {}", ndc.z);
    }

    #[test]
    fn orbit_changes_yaw_and_pitch_and_clamps_pitch() {
        let mut cam = OrbitCamera::default();
        let before = cam;
        cam.orbit(20.0, 10.0);
        assert_ne!(cam.yaw, before.yaw);
        assert_ne!(cam.pitch, before.pitch);
        cam.orbit(0.0, -100_000.0);
        assert_eq!(cam.pitch, MAX_PITCH);
        cam.orbit(0.0, 100_000.0);
        assert_eq!(cam.pitch, -MAX_PITCH);
    }

    #[test]
    fn zoom_moves_closer_on_positive_delta_and_clamps() {
        let mut cam = OrbitCamera::default();
        let d0 = cam.distance;
        cam.zoom(120.0);
        assert!(cam.distance < d0);
        cam.zoom(1.0e6);
        assert_eq!(cam.distance, MIN_DISTANCE);
        cam.zoom(-1.0e6);
        assert_eq!(cam.distance, MAX_DISTANCE);
    }

    #[test]
    fn pan_moves_target_in_the_view_plane() {
        let mut cam = OrbitCamera::default();
        let back = cam.back_direction();
        cam.pan(30.0, -12.0, 600.0);
        assert!(cam.target.length() > 0.0);
        // Panning never moves the target along the view direction.
        assert!(cam.target.dot(back).abs() < 1e-5);
        // A degenerate viewport is a no-op, not a NaN.
        let before = cam.target;
        cam.pan(10.0, 10.0, 0.0);
        assert_eq!(cam.target, before);
    }

    #[test]
    fn degenerate_aspect_falls_back_to_square() {
        let cam = OrbitCamera::default();
        assert_eq!(cam.projection(0.0), cam.projection(1.0));
        assert_eq!(cam.projection(f32::NAN), cam.projection(1.0));
    }
}
