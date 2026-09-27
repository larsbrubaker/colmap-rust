// wgpu side of the 3D viewport: a line-list pipeline that draws the `Scene`'s layers with the
// `OrbitCamera`, injected into agg-gui's frame through agg-gui-wgpu's custom-render hook
// (`WgpuCustomRender`, see agg-gui-wgpu/src/custom_render.rs). `viewport.rs` owns the widget
// and pushes this renderer each paint; GPU objects are created lazily on the first render and
// rebuilt when the device or surface format changes (device loss, backend switch).
//
// Draws straight into the active target (surface or layer) with a viewport and scissor set to
// the widget's rect and `LoadOp::Load`, so the widget's 2D background shows underneath. No depth
// buffer: the scene is lines only, drawn in layer order. Same approach as agg-gui's demo bar
// grid, minus its offscreen SSAA framebuffer.

use std::cell::RefCell;
use std::rc::Rc;

use agg_gui::Rect;
use agg_gui_wgpu::{WgpuCustomRender, WgpuCustomRenderCtx};
use wgpu::util::DeviceExt;

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
/// state as of `end_frame`.
pub struct ViewportRenderer {
    scene: Rc<RefCell<Scene>>,
    camera: Rc<RefCell<OrbitCamera>>,
    gpu: Option<LineGpu>,
}

impl ViewportRenderer {
    pub fn new(scene: Rc<RefCell<Scene>>, camera: Rc<RefCell<OrbitCamera>>) -> Self {
        Self {
            scene,
            camera,
            gpu: None,
        }
    }
}

impl WgpuCustomRender for ViewportRenderer {
    fn render(&mut self, ctx: WgpuCustomRenderCtx<'_>) {
        let Some(vp) = top_down_pixel_rect(ctx.screen_rect, ctx.target_size) else {
            return;
        };
        let scissor = match ctx.parent_clip {
            Some(clip) => match intersect(vp, clip_to_top_down(clip, ctx.target_size.1)) {
                Some(s) => s,
                None => return,
            },
            None => vp,
        };
        let stale = self
            .gpu
            .as_ref()
            .is_none_or(|g| g.device != *ctx.device || g.format != ctx.surface_format);
        if stale {
            self.gpu = Some(LineGpu::new(ctx.device, ctx.surface_format));
        }
        let Some(gpu) = self.gpu.as_mut() else {
            return;
        };
        gpu.sync_scene(&self.scene.borrow());
        let Some(vertices) = gpu.vertices.as_ref() else {
            return;
        };

        let aspect = vp[2] as f32 / vp[3].max(1) as f32;
        let view_proj = self.camera.borrow().view_projection(aspect).to_cols_array();
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

    #[test]
    fn scissor_intersection() {
        assert_eq!(
            intersect([0, 0, 10, 10], [5, 5, 10, 10]),
            Some([5, 5, 5, 5])
        );
        assert_eq!(intersect([0, 0, 10, 10], [20, 20, 5, 5]), None);
    }
}
