//! MONKEY (post): zone/day-night colour grading through a real 32³ GPU texture.
//!
//! Ported from WarcraftXL's `wxl-retail-grading` (Copyright (C) 2026 WarcraftXL, GPL-3.0-or-later).
//! Its 1024×32 BLP strip/two-bilinear-tap cube lookup becomes the equivalent hardware-trilinear
//! lookup after upload to a 3D texture. See `THIRD-PARTY.md`.

use super::bloom::BloomLabel;
use crate::video::VideoConfig;
use benilla_assets::{AssetSet, LockRecover, WorldAssets};
use benilla_world::{lighting::WorldTime, terrain_stream::CurrentArea, view::WorldCamera};
use bevy::{
    asset::RenderAssetUsages,
    core_pipeline::{core_3d::graph::Core3d, FullscreenShader},
    ecs::query::QueryItem,
    image::Image,
    prelude::*,
    render::{
        diagnostic::RecordDiagnostics,
        extract_component::{ExtractComponent, ExtractComponentPlugin},
        render_asset::RenderAssets,
        render_graph::{
            NodeRunError, RenderGraphContext, RenderGraphExt, RenderLabel, ViewNode, ViewNodeRunner,
        },
        render_resource::{binding_types::*, *},
        renderer::{RenderContext, RenderDevice, RenderQueue},
        texture::GpuImage,
        view::ViewTarget,
        RenderApp, RenderStartup,
    },
};
use std::collections::{HashMap, HashSet};

pub(super) struct GradingPlugin;

#[derive(Resource)]
struct GradeCatalog {
    rows: benilla_formats::MonkeyZoneGrades,
    luts: HashMap<String, Handle<Image>>,
    identity: Handle<Image>,
}

#[derive(Component, Clone, PartialEq, ExtractComponent)]
struct GradeView {
    day: Handle<Image>,
    night: Handle<Image>,
    // x = authored strength, y = night blend.
    control: Vec4,
}

#[derive(Clone, Copy, ShaderType)]
struct GradeUniform {
    control: Vec4,
}

impl Plugin for GradingPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(ExtractComponentPlugin::<GradeView>::default())
            .add_systems(Last, update_views);
        if !app.is_plugin_added::<AssetPlugin>() {
            return;
        }
        app.add_systems(Startup, load_catalog.after(AssetSet::Open));
        let shader = app
            .world_mut()
            .resource_mut::<Assets<Shader>>()
            .add(Shader::from_wgsl(
                include_str!("grading.wgsl"),
                "post/grading.wgsl",
            ));
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        render_app
            .insert_resource(GradeShader(shader))
            .add_systems(RenderStartup, init_pipeline)
            .add_render_graph_node::<ViewNodeRunner<GradeNode>>(Core3d, GradeLabel)
            .add_render_graph_edges(
                Core3d,
                (
                    BloomLabel,
                    GradeLabel,
                    benilla_world::ffx_glow::FfxGlowLabel,
                ),
            );
    }
}

fn load_catalog(
    mut commands: Commands,
    assets: Option<Res<WorldAssets>>,
    mut images: ResMut<Assets<Image>>,
) {
    let identity = images.add(volume_image(identity_cube()));
    let Some(assets) = assets else {
        commands.insert_resource(GradeCatalog {
            rows: default(),
            luts: default(),
            identity,
        });
        return;
    };
    let mut chain = assets.chain.lock_recover();
    let loose_root = benilla_formats::wow_data();
    let loaded_rows = loose_root
        .as_ref()
        .map(|root| root.join("DBFilesClient").join("MonkeyZoneGrade.dbc"))
        .filter(|path| path.is_file())
        .map(|path| {
            std::fs::read(path)
                .map_err(anyhow::Error::from)
                .and_then(|bytes| benilla_formats::parse_monkey_zone_grades(&bytes))
        })
        .unwrap_or_else(|| benilla_formats::load_monkey_zone_grades(&mut chain));
    let rows = match loaded_rows {
        Ok(rows) => rows,
        Err(error) => {
            info!("post grading: no MonkeyZoneGrade.dbc; identity fallback ({error:#})");
            commands.insert_resource(GradeCatalog {
                rows: default(),
                luts: default(),
                identity,
            });
            return;
        }
    };
    let paths: HashSet<String> = rows
        .rows()
        .flat_map(|row| [row.day_lut.clone(), row.night_lut.clone()])
        .collect();
    let mut luts = HashMap::new();
    for path in paths {
        let loose = loose_root
            .as_ref()
            .map(|root| root.join(path.replace('\\', "/")))
            .filter(|path| path.is_file())
            .map(std::fs::read)
            .transpose()
            .map_err(anyhow::Error::from);
        let result = loose
            .and_then(|bytes| match bytes {
                Some(bytes) => Ok(bytes),
                None => chain.read_file(&path.replace('/', "\\")),
            })
            .and_then(|bytes| benilla_formats::blp_bytes_to_mip_chain(&bytes));
        match result.and_then(|blp| strip_to_cube(&blp)) {
            Ok(cube) => {
                luts.insert(path, images.add(volume_image(cube)));
            }
            Err(error) => warn!("post grading: {path} rejected; identity fallback: {error:#}"),
        }
    }
    info!(
        "post grading: {} area rows, {} resident LUTs",
        rows.rows().count(),
        luts.len()
    );
    commands.insert_resource(GradeCatalog {
        rows,
        luts,
        identity,
    });
}

fn strip_to_cube(blp: &benilla_formats::BlpMipChain) -> anyhow::Result<Vec<u8>> {
    anyhow::ensure!(
        blp.width == 1024 && blp.height == 32,
        "expected 1024x32, got {}x{}",
        blp.width,
        blp.height
    );
    anyhow::ensure!(!blp.mips.is_empty(), "BLP has no base mip");
    let strip = &blp.mips[0];
    anyhow::ensure!(
        strip.len() == 1024 * 32 * 4,
        "base mip has {} bytes",
        strip.len()
    );
    let mut cube = vec![0; 32 * 32 * 32 * 4];
    for b in 0..32usize {
        for g in 0..32usize {
            for r in 0..32usize {
                let source = (g * 1024 + b * 32 + r) * 4;
                let destination = ((b * 32 + g) * 32 + r) * 4;
                cube[destination..destination + 4].copy_from_slice(&strip[source..source + 4]);
            }
        }
    }
    Ok(cube)
}

fn identity_cube() -> Vec<u8> {
    let mut cube = Vec::with_capacity(32 * 32 * 32 * 4);
    for b in 0..32u32 {
        for g in 0..32u32 {
            for r in 0..32u32 {
                cube.extend([r, g, b].map(|v| ((v * 255 + 15) / 31) as u8));
                cube.push(255);
            }
        }
    }
    cube
}

fn volume_image(cube: Vec<u8>) -> Image {
    Image::new(
        Extent3d {
            width: 32,
            height: 32,
            depth_or_array_layers: 32,
        },
        TextureDimension::D3,
        cube,
        TextureFormat::Rgba8Unorm,
        RenderAssetUsages::RENDER_WORLD,
    )
}

fn update_views(
    mut commands: Commands,
    video: Res<VideoConfig>,
    catalog: Option<Res<GradeCatalog>>,
    area: Res<CurrentArea>,
    areas: Option<Res<crate::area::AreaTableRes>>,
    time: Res<WorldTime>,
    cameras: Query<(Entity, &Camera, Option<&GradeView>), With<WorldCamera>>,
) {
    let selected = catalog.as_ref().and_then(|catalog| {
        let leaf = area.0?;
        let zone = areas
            .as_ref()
            .and_then(|areas| areas.0.top_zone(leaf))
            .unwrap_or(leaf);
        let row = catalog
            .rows
            .for_area(leaf)
            .or_else(|| catalog.rows.for_area(zone))?;
        (row.strength > 0.0).then(|| GradeView {
            day: catalog
                .luts
                .get(&row.day_lut)
                .unwrap_or(&catalog.identity)
                .clone(),
            night: catalog
                .luts
                .get(&row.night_lut)
                .unwrap_or(&catalog.identity)
                .clone(),
            control: Vec4::new(row.strength, night_weight(time.minute_f), 0.0, 0.0),
        })
    });
    for (entity, camera, old) in &cameras {
        let next = (video.color_grading && camera.is_active)
            .then(|| selected.clone())
            .flatten();
        match next {
            Some(next) if old != Some(&next) => {
                commands.entity(entity).insert(next);
            }
            None if old.is_some() => {
                commands.entity(entity).remove::<GradeView>();
            }
            _ => {}
        }
    }
}

fn night_weight(minute: f32) -> f32 {
    let dawn = smoothstep(330.0, 450.0, minute);
    let dusk = smoothstep(1110.0, 1230.0, minute);
    1.0 - dawn * (1.0 - dusk)
}

fn smoothstep(a: f32, b: f32, value: f32) -> f32 {
    let t = ((value - a) / (b - a)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

#[derive(Resource)]
struct GradeShader(Handle<Shader>);

#[derive(Resource)]
struct GradePipeline {
    layout: BindGroupLayoutDescriptor,
    scene_sampler: Sampler,
    lut_sampler: Sampler,
    pipeline: CachedRenderPipelineId,
}

#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
struct GradeLabel;

fn init_pipeline(
    mut commands: Commands,
    device: Res<RenderDevice>,
    cache: Res<PipelineCache>,
    shader: Res<GradeShader>,
    fullscreen: Res<FullscreenShader>,
) {
    let layout = BindGroupLayoutDescriptor::new(
        "post_grading_layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::FRAGMENT,
            (
                texture_2d(TextureSampleType::Float { filterable: true }),
                sampler(SamplerBindingType::Filtering),
                texture_3d(TextureSampleType::Float { filterable: true }),
                texture_3d(TextureSampleType::Float { filterable: true }),
                sampler(SamplerBindingType::Filtering),
                uniform_buffer::<GradeUniform>(false),
            ),
        ),
    );
    let pipeline = cache.queue_render_pipeline(RenderPipelineDescriptor {
        label: Some("post_grading".into()),
        layout: vec![layout.clone()],
        vertex: fullscreen.to_vertex_state(),
        fragment: Some(FragmentState {
            shader: shader.0.clone(),
            shader_defs: vec![],
            entry_point: Some("fragment".into()),
            targets: vec![Some(ColorTargetState {
                format: ViewTarget::TEXTURE_FORMAT_HDR,
                blend: None,
                write_mask: ColorWrites::ALL,
            })],
        }),
        ..default()
    });
    let sampler = |filter| {
        device.create_sampler(&SamplerDescriptor {
            min_filter: filter,
            mag_filter: filter,
            address_mode_u: AddressMode::ClampToEdge,
            address_mode_v: AddressMode::ClampToEdge,
            address_mode_w: AddressMode::ClampToEdge,
            ..default()
        })
    };
    commands.insert_resource(GradePipeline {
        layout,
        scene_sampler: sampler(FilterMode::Nearest),
        lut_sampler: sampler(FilterMode::Linear),
        pipeline,
    });
}

#[derive(Default)]
struct GradeNode;

impl ViewNode for GradeNode {
    type ViewQuery = (&'static ViewTarget, &'static GradeView);

    fn run<'w>(
        &self,
        _graph: &mut RenderGraphContext,
        context: &mut RenderContext<'w>,
        (target, grade): QueryItem<'w, '_, Self::ViewQuery>,
        world: &'w World,
    ) -> Result<(), NodeRunError> {
        let settings = world.resource::<GradePipeline>();
        let cache = world.resource::<PipelineCache>();
        let Some(pipeline) = cache.get_render_pipeline(settings.pipeline) else {
            return Ok(());
        };
        let images = world.resource::<RenderAssets<GpuImage>>();
        let (Some(day), Some(night)) = (images.get(&grade.day), images.get(&grade.night)) else {
            return Ok(());
        };
        let device = context.render_device();
        let mut uniform = UniformBuffer::from(GradeUniform {
            control: grade.control,
        });
        uniform.write_buffer(device, world.resource::<RenderQueue>());
        let out = target.post_process_write();
        let bind = device.create_bind_group(
            "post_grading",
            &cache.get_bind_group_layout(&settings.layout),
            &BindGroupEntries::sequential((
                out.source,
                &settings.scene_sampler,
                &day.texture_view,
                &night.texture_view,
                &settings.lut_sampler,
                uniform.binding().unwrap(),
            )),
        );
        let diagnostics = context.diagnostic_recorder();
        let mut pass = context
            .command_encoder()
            .begin_render_pass(&RenderPassDescriptor {
                label: Some("post_grading"),
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
        let span = diagnostics.pass_span(&mut pass, "post_grading");
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
    fn identity_volume_has_exact_axes() {
        let cube = identity_cube();
        let at = |r: usize, g: usize, b: usize| &cube[((b * 32 + g) * 32 + r) * 4..][..4];
        assert_eq!(at(0, 0, 0), &[0, 0, 0, 255]);
        assert_eq!(at(31, 0, 0), &[255, 0, 0, 255]);
        assert_eq!(at(0, 31, 0), &[0, 255, 0, 255]);
        assert_eq!(at(0, 0, 31), &[0, 0, 255, 255]);
    }

    #[test]
    fn day_night_blend_is_continuous() {
        assert_eq!(night_weight(0.0), 1.0);
        assert_eq!(night_weight(720.0), 0.0);
        assert_eq!(night_weight(1439.0), 1.0);
        assert!((night_weight(390.0) - 0.5).abs() < 1.0e-6);
    }
}
