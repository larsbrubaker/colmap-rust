// The 3D viewport's scene: an ordered list of drawable layers that each emit colored line
// segments in world space. Today the layers are the ground grid and the world-axis gizmo; later
// phases add camera frusta and point clouds as further layers. CPU-only (no wgpu), so the
// geometry is unit-tested here and `viewport_render.rs` just uploads what `Scene` produces.
//
// Not a COLMAP port (COLMAP's Qt/OpenGL viewer is excluded); world is Z-up as in `camera.rs`.

use bytemuck::{Pod, Zeroable};

/// One end of a line segment: world position and linear RGBA color. The layout is the vertex
/// buffer layout of the viewport's line pipeline (`viewport_render.rs`).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub struct LineVertex {
    pub position: [f32; 3],
    pub color: [f32; 4],
}

/// Something the viewport draws. Layers append line-list vertices (pairs) in world space.
pub trait SceneLayer {
    /// Stable name for diagnostics and tests.
    fn name(&self) -> &'static str;
    /// Append this layer's segments to `out` as consecutive vertex pairs.
    fn append_lines(&self, out: &mut Vec<LineVertex>);
}

/// Ground grid in the XY plane, centered on the origin.
#[derive(Clone, Copy, Debug)]
pub struct GridLayer {
    /// Number of cells from the origin to each edge.
    pub half_cells: u32,
    /// Cell size in world units.
    pub spacing: f32,
    pub color: [f32; 4],
    /// Color of every `major_every`-th line.
    pub major_color: [f32; 4],
    pub major_every: u32,
}

impl Default for GridLayer {
    fn default() -> Self {
        Self {
            half_cells: 10,
            spacing: 1.0,
            color: [0.28, 0.28, 0.30, 1.0],
            major_color: [0.42, 0.42, 0.45, 1.0],
            major_every: 5,
        }
    }
}

impl SceneLayer for GridLayer {
    fn name(&self) -> &'static str {
        "grid"
    }

    fn append_lines(&self, out: &mut Vec<LineVertex>) {
        let n = self.half_cells as i64;
        let extent = n as f32 * self.spacing;
        for i in -n..=n {
            let t = i as f32 * self.spacing;
            let major = self.major_every > 0 && i % self.major_every as i64 == 0;
            let color = if major { self.major_color } else { self.color };
            // Line parallel to X at y = t, and parallel to Y at x = t.
            out.push(LineVertex {
                position: [-extent, t, 0.0],
                color,
            });
            out.push(LineVertex {
                position: [extent, t, 0.0],
                color,
            });
            out.push(LineVertex {
                position: [t, -extent, 0.0],
                color,
            });
            out.push(LineVertex {
                position: [t, extent, 0.0],
                color,
            });
        }
    }
}

/// World-axis gizmo at the origin: X red, Y green, Z blue.
#[derive(Clone, Copy, Debug)]
pub struct AxisGizmoLayer {
    /// Axis length in world units.
    pub length: f32,
}

impl Default for AxisGizmoLayer {
    fn default() -> Self {
        Self { length: 3.0 }
    }
}

/// Axis colors, in X, Y, Z order.
pub const AXIS_COLORS: [[f32; 4]; 3] = [
    [0.95, 0.20, 0.20, 1.0],
    [0.25, 0.85, 0.25, 1.0],
    [0.25, 0.45, 1.00, 1.0],
];

impl SceneLayer for AxisGizmoLayer {
    fn name(&self) -> &'static str {
        "axis-gizmo"
    }

    fn append_lines(&self, out: &mut Vec<LineVertex>) {
        for (axis, color) in AXIS_COLORS.iter().enumerate() {
            let mut end = [0.0; 3];
            end[axis] = self.length;
            out.push(LineVertex {
                position: [0.0; 3],
                color: *color,
            });
            out.push(LineVertex {
                position: end,
                color: *color,
            });
        }
    }
}

/// The viewport's layers, drawn in order (later layers paint over earlier ones; there is no
/// depth buffer yet because lines alone do not need one).
pub struct Scene {
    layers: Vec<Box<dyn SceneLayer>>,
    /// Bumped on every change so the GPU side re-uploads only when needed.
    revision: u64,
}

impl Default for Scene {
    /// The Phase 0 scene: ground grid under the axis gizmo.
    fn default() -> Self {
        let mut scene = Self::empty();
        scene.push_layer(Box::new(GridLayer::default()));
        scene.push_layer(Box::new(AxisGizmoLayer::default()));
        scene
    }
}

impl Scene {
    /// A scene with no layers.
    pub fn empty() -> Self {
        Self {
            layers: Vec::new(),
            revision: 0,
        }
    }

    /// Append a layer on top of the existing ones.
    pub fn push_layer(&mut self, layer: Box<dyn SceneLayer>) {
        self.layers.push(layer);
        self.revision += 1;
    }

    /// Layer names, bottom to top.
    pub fn layer_names(&self) -> Vec<&'static str> {
        self.layers.iter().map(|l| l.name()).collect()
    }

    /// Changes whenever the layer list changes.
    pub fn revision(&self) -> u64 {
        self.revision
    }

    /// All layers' segments as one line list, in draw order.
    pub fn line_vertices(&self) -> Vec<LineVertex> {
        let mut out = Vec::new();
        for layer in &self.layers {
            layer.append_lines(&mut out);
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_scene_is_grid_then_gizmo() {
        let scene = Scene::default();
        assert_eq!(scene.layer_names(), vec!["grid", "axis-gizmo"]);
        assert_eq!(scene.line_vertices().len() % 2, 0);
    }

    #[test]
    fn gizmo_axes_are_red_green_blue_along_xyz() {
        let mut out = Vec::new();
        AxisGizmoLayer { length: 2.0 }.append_lines(&mut out);
        assert_eq!(out.len(), 6);
        assert_eq!(out[1].position, [2.0, 0.0, 0.0]);
        assert_eq!(out[3].position, [0.0, 2.0, 0.0]);
        assert_eq!(out[5].position, [0.0, 0.0, 2.0]);
        assert!(out[0].color[0] > out[0].color[1] && out[0].color[0] > out[0].color[2]);
        assert!(out[2].color[1] > out[2].color[0] && out[2].color[1] > out[2].color[2]);
        assert!(out[4].color[2] > out[4].color[0] && out[4].color[2] > out[4].color[1]);
    }

    #[test]
    fn grid_lies_in_the_ground_plane() {
        let mut out = Vec::new();
        let grid = GridLayer {
            half_cells: 2,
            ..GridLayer::default()
        };
        grid.append_lines(&mut out);
        // 2 * half_cells + 1 lines in each direction, 2 vertices each.
        assert_eq!(out.len(), 5 * 2 * 2);
        assert!(out.iter().all(|v| v.position[2] == 0.0));
    }

    #[test]
    fn pushing_a_layer_bumps_the_revision() {
        let mut scene = Scene::empty();
        let r0 = scene.revision();
        scene.push_layer(Box::new(GridLayer::default()));
        assert!(scene.revision() > r0);
    }
}
