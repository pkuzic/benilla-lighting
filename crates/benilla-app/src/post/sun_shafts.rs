//! MONKEY (post): screen-space sun shafts from the visible sky depth.

use super::bloom::BloomLabel;
use crate::video::VideoConfig;
use benilla_world::{lighting::WowLighting, view::WorldCamera, wmo_portal::CameraInteriorClaim};
use bevy::{
    core_pipeline::{
        core_3d::graph::{Core3d, Node3d},
        FullscreenShader,
    },
    ecs::query::QueryItem,
    prelude::*,
    render::{
        diagnostic::RecordDiagnostics,
        extract_component::{ExtractComponent, ExtractComponentPlugin},
        render_graph::{
            NodeRunError, RenderGraphContext, RenderGraphExt, RenderLabel, ViewNode, ViewNodeRunner,
        },
        render_resource::{binding_types::*, *},
        renderer::{RenderContext, RenderDevice, RenderQueue},
        view::{Msaa, ViewDepthTexture, ViewTarget},
        Render, RenderApp, RenderStartup, RenderSystems,
    },
};

pub(super) struct SunShaftsPlugin;

#[derive(Component, Clone, Copy, PartialEq, ExtractComponent, ShaderType)]
struct ShaftView {
    // xy = sun UV, z = faded strength, w = sample count.
    sun: Vec4,
    // Gamma-space sun colour; the framebuffer is still gamma-space here.
    color: Vec4,
}

impl Plugin for SunShaftsPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(ExtractComponentPlugin::<ShaftView>::default())
            .add_systems(Last, update_views);
        if !app.is_plugin_added::<AssetPlugin>() {
            return;
        }
        let shader = app
            .world_mut()
            .resource_mut::<Assets<Shader>>()
            .add(Shader::from_wgsl(
                include_str!("sun_shafts.wgsl"),
                "post/sun_shafts.wgsl",
            ));
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        render_app
            .insert_resource(ShaftShader(shader))
            .init_resource::<SpecializedRenderPipelines<ShaftPipeline>>()
            .add_systems(RenderStartup, init_pipeline)
            .add_systems(Render, prepare_pipelines.in_set(RenderSystems::Prepare))
            .add_render_graph_node::<ViewNodeRunner<ShaftNode>>(Core3d, ShaftLabel)
            .add_render_graph_edges(
                Core3d,
                (Node3d::StartMainPassPostProcessing, ShaftLabel, BloomLabel),
            );
    }
}

fn update_views(
    mut commands: Commands,
    video: Res<VideoConfig>,
    lighting: Res<WowLighting>,
    interior: Res<CameraInteriorClaim>,
    mut cameras: Query<
        (
            Entity,
            &Camera,
            &GlobalTransform,
            &Projection,
            &mut Camera3d,
            Option<&ShaftView>,
        ),
        With<WorldCamera>,
    >,
) {
    let to_sun = lighting.celestial_dir().normalize_or_zero();
    for (entity, camera, transform, projection, mut camera3d, old) in &mut cameras {
        let disabled =
            !video.sun_shafts || !camera.is_active || interior.0.is_some() || to_sun == Vec3::ZERO;
        if disabled {
            if old.is_some() {
                commands.entity(entity).remove::<ShaftView>();
            }
            continue;
        }
        let far = match projection {
            Projection::Perspective(p) => p.far,
            _ => 3000.0,
        };
        let world = transform.translation() + to_sun * far * 0.85;
        let (Ok(screen), Some(size)) = (
            camera.world_to_viewport(transform, world),
            camera.logical_viewport_size(),
        ) else {
            commands.entity(entity).remove::<ShaftView>();
            continue;
        };
        let uv = screen / size;
        // Keep a small off-screen skirt so shafts leave the frame continuously.
        let outside = Vec2::new(
            (-uv.x).max(uv.x - 1.0).max(0.0),
            (-uv.y).max(uv.y - 1.0).max(0.0),
        )
        .max_element();
        let edge_fade = 1.0 - smoothstep(0.0, 0.18, outside);
        let daylight = smoothstep(-0.02, 0.16, to_sun.y);
        let strength = edge_fade * daylight;
        if strength <= 0.001 {
            commands.entity(entity).remove::<ShaftView>();
            continue;
        }
        camera3d.depth_texture_usages.0 |= TextureUsages::TEXTURE_BINDING.bits();
        let next = ShaftView {
            sun: uv.extend(strength).extend(28.0),
            color: Vec3::from_array(lighting.diffuse)
                .lerp(Vec3::ONE, 0.18)
                .extend(0.0),
        };
        if old != Some(&next) {
            commands.entity(entity).insert(next);
        }
    }
}

fn smoothstep(a: f32, b: f32, value: f32) -> f32 {
    let t = ((value - a) / (b - a)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

#[derive(Resource)]
struct ShaftShader(Handle<Shader>);

#[derive(Resource)]
struct ShaftPipeline {
    layouts: [BindGroupLayoutDescriptor; 2],
    shader: Handle<Shader>,
    fullscreen: FullscreenShader,
    sampler: Sampler,
}

#[derive(Component)]
struct ViewShaftPipeline(CachedRenderPipelineId);

#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
struct ShaftLabel;

fn init_pipeline(
    mut commands: Commands,
    device: Res<RenderDevice>,
    shader: Res<ShaftShader>,
    fullscreen: Res<FullscreenShader>,
) {
    let layouts = [false, true].map(|multisampled| {
        BindGroupLayoutDescriptor::new(
            "post_sun_shafts_layout",
            &BindGroupLayoutEntries::sequential(
                ShaderStages::FRAGMENT,
                (
                    texture_2d(TextureSampleType::Float { filterable: true }),
                    sampler(SamplerBindingType::Filtering),
                    if multisampled {
                        texture_depth_2d_multisampled()
                    } else {
                        texture_depth_2d()
                    },
                    uniform_buffer::<ShaftView>(false),
                ),
            ),
        )
    });
    commands.insert_resource(ShaftPipeline {
        layouts,
        shader: shader.0.clone(),
        fullscreen: fullscreen.clone(),
        sampler: device.create_sampler(&SamplerDescriptor {
            min_filter: FilterMode::Linear,
            mag_filter: FilterMode::Linear,
            address_mode_u: AddressMode::ClampToEdge,
            address_mode_v: AddressMode::ClampToEdge,
            ..default()
        }),
    });
}

impl SpecializedRenderPipeline for ShaftPipeline {
    type Key = (TextureFormat, bool);

    fn specialize(&self, (format, multisampled): Self::Key) -> RenderPipelineDescriptor {
        RenderPipelineDescriptor {
            label: Some("post_sun_shafts".into()),
            layout: vec![self.layouts[multisampled as usize].clone()],
            vertex: self.fullscreen.to_vertex_state(),
            fragment: Some(FragmentState {
                shader: self.shader.clone(),
                shader_defs: multisampled
                    .then(|| "MULTISAMPLED".into())
                    .into_iter()
                    .collect(),
                entry_point: Some("fragment".into()),
                targets: vec![Some(ColorTargetState {
                    format,
                    blend: None,
                    write_mask: ColorWrites::ALL,
                })],
            }),
            ..default()
        }
    }
}

fn prepare_pipelines(
    mut commands: Commands,
    cache: Res<PipelineCache>,
    pipeline: Res<ShaftPipeline>,
    mut specialized: ResMut<SpecializedRenderPipelines<ShaftPipeline>>,
    views: Query<(Entity, &ViewTarget, &Msaa), With<ShaftView>>,
) {
    for (entity, target, msaa) in &views {
        let id = specialized.specialize(
            &cache,
            &pipeline,
            (target.main_texture_format(), msaa.samples() > 1),
        );
        commands.entity(entity).insert(ViewShaftPipeline(id));
    }
}

#[derive(Default)]
struct ShaftNode;

impl ViewNode for ShaftNode {
    type ViewQuery = (
        &'static ViewTarget,
        &'static ViewDepthTexture,
        &'static ShaftView,
        &'static ViewShaftPipeline,
    );

    fn run<'w>(
        &self,
        _graph: &mut RenderGraphContext,
        context: &mut RenderContext<'w>,
        (target, depth, shaft, id): QueryItem<'w, '_, Self::ViewQuery>,
        world: &'w World,
    ) -> Result<(), NodeRunError> {
        let cache = world.resource::<PipelineCache>();
        let Some(pipeline) = cache.get_render_pipeline(id.0) else {
            return Ok(());
        };
        if !depth
            .texture
            .usage()
            .contains(TextureUsages::TEXTURE_BINDING)
        {
            return Ok(());
        }
        let settings = world.resource::<ShaftPipeline>();
        let device = context.render_device();
        let mut uniform = UniformBuffer::from(*shaft);
        uniform.write_buffer(device, world.resource::<RenderQueue>());
        let out = target.post_process_write();
        let bind = device.create_bind_group(
            "post_sun_shafts",
            &cache.get_bind_group_layout(
                &settings.layouts[(depth.texture.sample_count() > 1) as usize],
            ),
            &BindGroupEntries::sequential((
                out.source,
                &settings.sampler,
                depth.view(),
                uniform.binding().unwrap(),
            )),
        );
        let diagnostics = context.diagnostic_recorder();
        let mut pass = context
            .command_encoder()
            .begin_render_pass(&RenderPassDescriptor {
                label: Some("post_sun_shafts"),
                color_attachments: &[Some(RenderPassColorAttachment {
                    view: out.destination,
                    depth_slice: None,
                    resolve_target: None,
                    ops: Operations::default(),
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
        let span = diagnostics.pass_span(&mut pass, "post_sun_shafts");
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind, &[]);
        pass.draw(0..3, 0..1);
        span.end(&mut pass);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn edge_and_night_fades_are_bounded() {
        assert_eq!(smoothstep(0.0, 1.0, -1.0), 0.0);
        assert_eq!(smoothstep(0.0, 1.0, 2.0), 1.0);
        assert!((smoothstep(0.0, 1.0, 0.5) - 0.5).abs() < 1.0e-6);
    }
}
