// wgpu side of the 3D viewport: a line-list pipeline that draws the `Scene`'s layers with the
// `OrbitCamera`, injected into agg-gui's frame through agg-gui-wgpu's custom-render hook
// (`WgpuCustomRender`, see agg-gui-wgpu/src/custom_render.rs). `viewport.rs` owns the widget
// and pushes this renderer each paint; GPU objects are created lazily on the first render and
// rebuilt when the device or surface format changes (device loss, backend switch). Each rebuild
// republishes the adapter into `AppState::backend`, so the About sheet names the GPU in use.
//
// Draws straight into the active target (surface or layer) with `LoadOp::Load`, so the widget's
// 2D background shows underneath. The projection always covers the widget's full (unclipped)
// rect, so a widget partly outside the target or its parent clip is cut off, never squashed:
// the wgpu viewport is the rect clipped to the target (WebGPU rejects viewports that leave the
// attachment), a clip-space correction maps the full rect's projection onto it, and parent
// clipping only narrows the scissor (`ViewportLayout`). No depth
// buffer: the scene is lines only, drawn in layer order. Same approach as agg-gui's demo bar
// grid, minus its offscreen SSAA framebuffer.

use std::cell::RefCell;
use std::rc::Rc;

use agg_gui::Rect;
use agg_gui_wgpu::{WgpuCustomRender, WgpuCustomRenderCtx};
use wgpu::util::DeviceExt;

use crate::backend_info::{backend_info, BackendInfo};
use crate::camera::OrbitCamera;
use crate::scene::{LineVertex, Scene};

const LINE_WGSL: &str = "
struct Uniforms { view_proj: mat4x4<f32> }
@group(0) @binding(0) var<uniform> u: Uniforms;

struct VIn {
    @location(0) pos: vec3<f32>,
    @location(1) color: vec4<f32>,
}
struct VOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) color: vec4<f32>,
}

@vertex fn vs_main(v: VIn) -> VOut {
    var out: VOut;
    out.clip = u.view_proj * vec4<f32>(v.pos, 1.0);
    out.color = v.color;
    return out;
}

@fragment fn fs_main(v: VOut) -> @location(0) vec4<f32> {
    return v.color;
}
";

/// Size of the uniform block: one column-major 4x4 f32 matrix.
const UNIFORM_BYTES: u64 = 64;

/// GPU objects tied to one device + target format.
struct LineGpu {
    device: wgpu::Device,
    format: wgpu::TextureFormat,
    pipeline: wgpu::RenderPipeline,
    uniforms: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    vertices: Option<wgpu::Buffer>,
    vertex_count: u32,
    /// `Scene::revision` the vertex buffer was built from.
    scene_revision: Option<u64>,
}

impl LineGpu {
    fn new(device: &wgpu::Device, format: wgpu::TextureFormat) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("colmap_viewport_lines"),
            source: wgpu::ShaderSource::Wgsl(LINE_WGSL.into()),
        });
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("colmap_viewport_bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let layouts = [Some(&bgl)];
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("colmap_viewport_layout"),
            bind_group_layouts: &layouts,
            immediate_size: 0,
        });
        let attributes = [
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x3,
                offset: 0,
                shader_location: 0,
            },
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x4,
                offset: 12,
                shader_location: 1,
            },
        ];
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("colmap_viewport_pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<LineVertex>() as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &attributes,
                }],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::LineList,
                ..Default::default()
            },
            depth_stencil: None,
            // agg-gui-wgpu's 2D targets are single-sample (its AA is SSAA / coverage based).
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });
        let uniforms = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("colmap_viewport_uniforms"),
            size: UNIFORM_BYTES,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("colmap_viewport_bg"),
            layout: &bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniforms.as_entire_binding(),
            }],
        });
        Self {
            device: device.clone(),
            format,
            pipeline,
            uniforms,
            bind_group,
            vertices: None,
            vertex_count: 0,
            scene_revision: None,
        }
    }

    /// Re-upload the scene's lines when its revision changed.
    fn sync_scene(&mut self, scene: &Scene) {
        if self.scene_revision == Some(scene.revision()) {
            return;
        }
        let lines = scene.line_vertices();
        self.vertex_count = lines.len() as u32;
        self.vertices = if lines.is_empty() {
            None
        } else {
            Some(
                self.device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("colmap_viewport_lines"),
                        contents: bytemuck::cast_slice(&lines),
                        usage: wgpu::BufferUsages::VERTEX,
                    }),
            )
        };
        self.scene_revision = Some(scene.revision());
    }
}

/// The widget's full rect as `[x, y, w, h]` in top-down pixels, unclipped (it may extend past
/// the target on any side).
pub fn full_top_down_rect(rect: Rect, target_h: u32) -> [f64; 4] {
    [
        rect.x,
        target_h as f64 - (rect.y + rect.height),
        rect.width,
        rect.height,
    ]
}

/// Where and how the viewport draws in one frame.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ViewportLayout {
    /// wgpu viewport: the widget rect clipped to the target, top-down pixels.
    pub viewport: [u32; 4],
    /// Scissor: the viewport further clipped by the parent clip.
    pub scissor: [u32; 4],
    /// Projection aspect: the full widget rect's width / height.
    pub aspect: f32,
    /// Clip-space correction `(sx, sy, tx, ty)`: `x' = sx*x + tx*w`, `y' = sy*y + ty*w`. It
    /// maps NDC over the full widget rect onto NDC over `viewport`, so the picture is the one a
    /// viewport covering the whole widget would give. Identity when nothing is clipped.
    pub correction: [f32; 4],
}

impl ViewportLayout {
    /// Layout for a Y-up widget `rect` on a `target`-sized attachment with agg-gui-wgpu's
    /// Y-up `parent_clip`. `None` when nothing is visible.
    pub fn new(rect: Rect, target: (u32, u32), parent_clip: Option<[i32; 4]>) -> Option<Self> {
        let viewport = top_down_pixel_rect(rect, target)?;
        let scissor = match parent_clip {
            Some(clip) => intersect(viewport, clip_to_top_down(clip, target.1))?,
            None => viewport,
        };
        let [fx, fy, fw, fh] = full_top_down_rect(rect, target.1);
        if fw <= 0.0 || fh <= 0.0 {
            return None;
        }
        let [vx, vy, vw, vh] = viewport.map(f64::from);
        // pixel_x = fx + (ndc_f + 1)/2 * fw = vx + (ndc_v + 1)/2 * vw, solved for ndc_v; same for
        // y with NDC pointing up and pixels pointing down.
        let sx = fw / vw;
        let tx = (2.0 * (fx - vx) + fw) / vw - 1.0;
        let sy = fh / vh;
        let ty = 1.0 - (2.0 * (fy - vy) + fh) / vh;
        Some(Self {
            viewport,
            scissor,
            aspect: (fw / fh) as f32,
            correction: [sx as f32, sy as f32, tx as f32, ty as f32],
        })
    }

    /// The correction as a matrix to left-multiply onto the view-projection.
    pub fn correction_matrix(&self) -> glam::Mat4 {
        let [sx, sy, tx, ty] = self.correction;
        glam::Mat4::from_cols(
            glam::Vec4::new(sx, 0.0, 0.0, 0.0),
            glam::Vec4::new(0.0, sy, 0.0, 0.0),
            glam::Vec4::new(0.0, 0.0, 1.0, 0.0),
            glam::Vec4::new(tx, ty, 0.0, 1.0),
        )
    }
}

/// Integer `[x, y, w, h]` in wgpu top-down pixels for a Y-up `rect` on a target `target_h`
/// pixels tall, clipped to the target. `None` when nothing is left.
pub fn top_down_pixel_rect(rect: Rect, target: (u32, u32)) -> Option<[u32; 4]> {
    let (tw, th) = (target.0 as f64, target.1 as f64);
    let x0 = rect.x.floor().max(0.0);
    let x1 = (rect.x + rect.width).ceil().min(tw);
    let y0 = (th - (rect.y + rect.height)).floor().max(0.0);
    let y1 = (th - rect.y).ceil().min(th);
    if x1 <= x0 || y1 <= y0 {
        return None;
    }
    Some([x0 as u32, y0 as u32, (x1 - x0) as u32, (y1 - y0) as u32])
}

/// agg-gui-wgpu's `parent_clip` as a top-down rect. The clip is stored Y-up
/// (`[x, y_bottom, w, h]`, see agg-gui-wgpu `ctx_core::compute_scissor`) even though the
/// `WgpuCustomRenderCtx` doc says top-down; treating it as top-down cut the viewport's bottom
/// strip by the height of the top bar (seen in the native app).
pub fn clip_to_top_down(clip: [i32; 4], target_h: u32) -> [i32; 4] {
    let [x, y_bottom, w, h] = clip;
    [x, target_h as i32 - (y_bottom + h), w, h]
}

/// Intersect two `[x, y, w, h]` rects.
fn intersect(a: [u32; 4], b: [i32; 4]) -> Option<[u32; 4]> {
    let bx0 = b[0].max(0) as u32;
    let by0 = b[1].max(0) as u32;
    let bx1 = (b[0] + b[2]).max(0) as u32;
    let by1 = (b[1] + b[3]).max(0) as u32;
    let x0 = a[0].max(bx0);
    let y0 = a[1].max(by0);
    let x1 = (a[0] + a[2]).min(bx1);
    let y1 = (a[1] + a[3]).min(by1);
    (x1 > x0 && y1 > y0).then(|| [x0, y0, x1 - x0, y1 - y0])
}

/// The viewport's custom renderer. Holds the shared scene and camera so the frame drawn is the
/// state as of `end_frame`, and the shared backend cell it republishes on every GPU rebuild.
pub struct ViewportRenderer {
    scene: Rc<RefCell<Scene>>,
    camera: Rc<RefCell<OrbitCamera>>,
    backend: Rc<RefCell<Option<BackendInfo>>>,
    gpu: Option<LineGpu>,
}

impl ViewportRenderer {
    pub fn new(
        scene: Rc<RefCell<Scene>>,
        camera: Rc<RefCell<OrbitCamera>>,
        backend: Rc<RefCell<Option<BackendInfo>>>,
    ) -> Self {
        Self {
            scene,
            camera,
            backend,
            gpu: None,
        }
    }

    /// (Re)build the GPU objects when the device or target format changed, publishing the
    /// device's adapter to the backend cell each time it does. Runs every frame, visible or
    /// not, so the About sheet is right even while the viewport is clipped away.
    ///
    /// Caveat: `wgpu::Device`'s `==` compares only the wgpu-core device id, which restarts per
    /// `wgpu::Instance`, so a device from a freshly created instance can compare equal to the
    /// old one and skip the rebuild.
    fn ensure_gpu(&mut self, device: &wgpu::Device, format: wgpu::TextureFormat) -> &mut LineGpu {
        let stale = self
            .gpu
            .as_ref()
            .is_none_or(|g| g.device != *device || g.format != format);
        if stale {
            *self.backend.borrow_mut() = Some(backend_info(&device.adapter_info()));
            self.gpu = None;
        }
        self.gpu.get_or_insert_with(|| LineGpu::new(device, format))
    }
}

impl WgpuCustomRender for ViewportRenderer {
    fn render(&mut self, ctx: WgpuCustomRenderCtx<'_>) {
        let (scene, camera) = (self.scene.clone(), self.camera.clone());
        let gpu = self.ensure_gpu(ctx.device, ctx.surface_format);
        let Some(layout) = ViewportLayout::new(ctx.screen_rect, ctx.target_size, ctx.parent_clip)
        else {
            return;
        };
        let (vp, scissor) = (layout.viewport, layout.scissor);
        gpu.sync_scene(&scene.borrow());
        let Some(vertices) = gpu.vertices.as_ref() else {
            return;
        };

        let view_proj = (layout.correction_matrix()
            * camera.borrow().view_projection(layout.aspect))
        .to_cols_array();
        ctx.queue
            .write_buffer(&gpu.uniforms, 0, bytemuck::cast_slice(&view_proj));

        let mut pass = ctx.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("colmap_viewport_pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: ctx.target_view,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_viewport(
            vp[0] as f32,
            vp[1] as f32,
            vp[2] as f32,
            vp[3] as f32,
            0.0,
            1.0,
        );
        pass.set_scissor_rect(scissor[0], scissor[1], scissor[2], scissor[3]);
        pass.set_pipeline(&gpu.pipeline);
        pass.set_bind_group(0, &gpu.bind_group, &[]);
        pass.set_vertex_buffer(0, vertices.slice(..));
        pass.draw(0..gpu.vertex_count, 0..1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn y_up_rect_flips_to_top_down() {
        // A 100x50 rect whose bottom edge is 10 px above the bottom of a 200 px target.
        let r = top_down_pixel_rect(Rect::new(20.0, 10.0, 100.0, 50.0), (400, 200));
        assert_eq!(r, Some([20, 140, 100, 50]));
    }

    #[test]
    fn rect_is_clipped_to_the_target() {
        let r = top_down_pixel_rect(Rect::new(-10.0, -10.0, 50.0, 50.0), (30, 30));
        assert_eq!(r, Some([0, 0, 30, 30]));
        assert_eq!(
            top_down_pixel_rect(Rect::new(500.0, 0.0, 10.0, 10.0), (30, 30)),
            None
        );
    }

    #[test]
    fn parent_clip_flips_to_top_down() {
        // A clip covering the bottom 700 px of an 800 px target starts 100 px from the top.
        assert_eq!(
            clip_to_top_down([0, 0, 1280, 700], 800),
            [0, 100, 1280, 700]
        );
    }

    /// Top-down pixel where clip-space `(x, y, w)` lands under `layout`'s viewport.
    fn to_pixel(layout: &ViewportLayout, x: f32, y: f32, w: f32) -> (f32, f32) {
        let clip = layout.correction_matrix() * glam::Vec4::new(x, y, 0.0, w);
        let (nx, ny) = (clip.x / clip.w, clip.y / clip.w);
        let [vx, vy, vw, vh] = layout.viewport.map(|v| v as f32);
        (vx + (nx + 1.0) / 2.0 * vw, vy + (1.0 - ny) / 2.0 * vh)
    }

    #[test]
    fn unclipped_layout_is_identity() {
        let layout =
            ViewportLayout::new(Rect::new(20.0, 10.0, 100.0, 50.0), (400, 200), None).unwrap();
        assert_eq!(layout.viewport, [20, 140, 100, 50]);
        assert_eq!(layout.scissor, layout.viewport);
        assert_eq!(layout.aspect, 2.0);
        assert_eq!(layout.correction, [1.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn partly_offscreen_widget_keeps_full_rect_projection() {
        // 200x100 widget hanging 50 px off the left and 40 px off the top of a 300x300 target.
        let rect = Rect::new(-50.0, 240.0, 200.0, 100.0);
        let layout = ViewportLayout::new(rect, (300, 300), None).unwrap();
        assert_eq!(layout.viewport, [0, 0, 150, 60]);
        assert_eq!(layout.aspect, 2.0);
        // The full rect's NDC corners land on the full rect's pixel corners, at any w.
        // (f32 clip math: within 1e-3 px.)
        let near =
            |a: (f32, f32), b: (f32, f32)| (a.0 - b.0).abs() < 1e-3 && (a.1 - b.1).abs() < 1e-3;
        for w in [1.0, 3.5] {
            assert!(near(to_pixel(&layout, -w, w, w), (-50.0, -40.0)));
            assert!(near(to_pixel(&layout, w, -w, w), (150.0, 60.0)));
            assert!(near(to_pixel(&layout, 0.0, 0.0, w), (50.0, 10.0)));
        }
    }

    #[test]
    fn parent_clip_narrows_only_the_scissor() {
        // A clip covering only the bottom 100 px of a 400x400 target.
        let rect = Rect::new(0.0, 0.0, 400.0, 200.0);
        let layout = ViewportLayout::new(rect, (400, 400), Some([0, 0, 400, 100])).unwrap();
        assert_eq!(layout.viewport, [0, 200, 400, 200]);
        assert_eq!(layout.scissor, [0, 300, 400, 100]);
        assert_eq!(layout.aspect, 2.0);
        assert_eq!(layout.correction, [1.0, 1.0, 0.0, 0.0]);
        assert_eq!(
            ViewportLayout::new(rect, (400, 400), Some([0, 300, 400, 100])),
            None
        );
    }

    /// Resolve a future that is already complete (the no-op backend's are).
    fn ready<F: std::future::Future>(f: F) -> F::Output {
        let mut cx = std::task::Context::from_waker(std::task::Waker::noop());
        match std::pin::pin!(f).poll(&mut cx) {
            std::task::Poll::Ready(v) => v,
            std::task::Poll::Pending => panic!("noop backend future pending"),
        }
    }

    /// `n` devices on one adapter of wgpu's no-op backend (no GPU needed). One instance on
    /// purpose: `wgpu::Device`'s `==` compares only wgpu-core ids, which restart per instance,
    /// so devices from two instances can compare equal.
    fn noop_devices(n: usize) -> Vec<wgpu::Device> {
        let mut desc = wgpu::InstanceDescriptor::new_without_display_handle();
        desc.backends = wgpu::Backends::NOOP;
        desc.backend_options.noop.enable = true;
        let instance = wgpu::Instance::new(desc);
        let adapter = ready(instance.request_adapter(&Default::default())).expect("noop adapter");
        (0..n)
            .map(|_| {
                ready(adapter.request_device(&Default::default()))
                    .expect("noop device")
                    .0
            })
            .collect()
    }

    #[test]
    fn gpu_rebuild_republishes_the_backend() {
        let backend = Rc::new(RefCell::new(None));
        let mut renderer =
            ViewportRenderer::new(Default::default(), Default::default(), backend.clone());
        let format = wgpu::TextureFormat::Bgra8Unorm;
        let devices = noop_devices(2);
        let (first, second) = (&devices[0], &devices[1]);
        assert!(first != second);
        renderer.ensure_gpu(first, format);
        let published = backend.borrow().clone().expect("published on first build");
        assert_eq!(published, backend_info(&first.adapter_info()));
        assert_eq!(published.backend, "no-op");

        // Same device and format: no rebuild, the cell is left alone.
        let sentinel = BackendInfo {
            adapter_name: "sentinel".into(),
            backend: "-".into(),
            device_type: "-".into(),
            driver: "-".into(),
        };
        *backend.borrow_mut() = Some(sentinel.clone());
        renderer.ensure_gpu(first, format);
        assert_eq!(backend.borrow().as_ref(), Some(&sentinel));

        // A new device (device loss / backend switch) rebuilds and republishes.
        renderer.ensure_gpu(second, format);
        assert_eq!(backend.borrow().as_ref(), Some(&published));
        // So does a new target format.
        *backend.borrow_mut() = Some(sentinel);
        renderer.ensure_gpu(second, wgpu::TextureFormat::Rgba8Unorm);
        assert_eq!(backend.borrow().as_ref(), Some(&published));
    }

    #[test]
    fn scissor_intersection() {
        assert_eq!(
            intersect([0, 0, 10, 10], [5, 5, 10, 10]),
            Some([5, 5, 5, 5])
        );
        assert_eq!(intersect([0, 0, 10, 10], [20, 20, 5, 5]), None);
    }
}
