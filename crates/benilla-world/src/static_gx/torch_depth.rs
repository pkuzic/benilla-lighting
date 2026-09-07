//! MONKEY (torch shadows, Phase 1): a from-scratch depth-map render for the nearest interior torch
//! fixtures.
//!
//! benilla's world camera sets `ClusterConfig::None`, which disables Bevy's point/spot clustering AND
//! starves the point-light shadow-map prep — so Bevy's built-in `fetch_point_shadow` path is dead for
//! us (see `benilla_app::torch_shadow`). This module replaces it: for each of the nearest ≤4 interior
//! fixtures (published by the app as [`TorchShadowViews`]) it renders the retained interior caster
//! geometry from the fixture's viewpoint into one layer of a `Depth32Float` 2D-array texture.
//! `static_gx.wgsl`'s interior surface lane then samples that array (group 3, built in `render.rs`)
//! so a pillar throws a radial shadow on the floor.
//!
//! **Phase 3A: the depth array is a shared `Image` ASSET**, not a render-world-owned texture. The
//! entity receiver (`wow_model.wgsl`, every unit/NPC/GameObject/interior prop) has no spare bind
//! group — a Bevy material draw sets groups 0/1/2 only — so its torch bindings ride the material's
//! own group as `Handle<Image>` + raw `Buffer` fields (`WowModelExt::torch_depth`/`torch_buf`),
//! and `AsBindGroup` resolves a `Handle<Image>` through `RenderAssets<GpuImage>`. So the image
//! ([`TorchDepthImage`]) is created ONCE at app startup in the main world (always — an absent image
//! would stall every model material's bind group and blank every model), extracted here, and BOTH
//! receivers read the same `GpuImage`: the depth node renders into per-layer views of its texture,
//! static_gx binds its D2Array view, the material binds the same view + the image's comparison
//! sampler. The ≤4-fixture table gets the same treatment ([`SharedTorchBuffer`]: one raw buffer,
//! rewritten in place every frame from [`TorchShadowViews`] with the SAME [`TorchTableUniform`]
//! bytes static_gx's group-3 uniform carries).
//!
//! Reverse-Z throughout (the whole engine is — `capture::depth_probe`): the depth pass clears to 0.0
//! (= far), keeps the nearest surface with `GreaterEqual`, and the comparison sampler is `GreaterEqual`
//! too. The per-fixture `view_proj` is a reverse-Z perspective built app-side in f64 (fixtures sit at
//! absolute Bevy world coordinates, ~9,300 out) and downcast to f32.

use bevy::asset::RenderAssetUsages;
use bevy::core_pipeline::core_3d::graph::{Core3d, Node3d};
use bevy::ecs::query::QueryItem;
use bevy::ecs::system::SystemParam;
use bevy::image::{ImageCompareFunction, ImageFilterMode, ImageSampler, ImageSamplerDescriptor};
// Bevy's OWNED vertex-buffer layout (the pipeline descriptor wants this, not wgpu's borrowed one).
use bevy::mesh::VertexBufferLayout;
use bevy::prelude::*;
use bevy::render::extract_resource::{ExtractResource, ExtractResourcePlugin};
use bevy::render::mesh::allocator::MeshAllocator;
use bevy::render::mesh::{RenderMesh, RenderMeshBufferInfo};
use bevy::render::render_asset::RenderAssets;
use bevy::render::render_graph::{
    NodeRunError, RenderGraphContext, RenderGraphExt, RenderLabel, ViewNode, ViewNodeRunner,
};
use bevy::render::render_resource::binding_types::uniform_buffer_sized;
use bevy::render::render_resource::*;
use bevy::render::renderer::{RenderContext, RenderDevice, RenderQueue};
use bevy::render::texture::GpuImage;
use bevy::render::{Render, RenderApp, RenderStartup, RenderSystems};
use bytemuck::Zeroable;

use super::render::StaticGxView;
use benilla_assets::materials::TorchBinds;

/// The depth-map edge (square) of ONE cube face. 512 keeps 24 faces at 24 MB of Depth32Float.
const TORCH_MAP_EDGE: u32 = 512;
/// The fixture cap — four fixtures, four positions.
pub(crate) const MAX_TORCH_MAPS: usize = 4;
/// MONKEY (Phase 5): six cube faces per fixture — a 90° reverse-Z perspective down each of ±X, ±Y,
/// ±Z — so the maps cover EVERY direction. The single aimed cone this replaces always had an edge,
/// and both the vertical (walk up to the forge) and lateral (stand beside the player) shadow
/// collapses were that edge. Layer index = `fixture * CUBE_FACES + face`; the face order is the
/// contract with `benilla_app::torch_shadow::cube_view_projs` and `static_gx.wgsl`'s `torch_face`.
pub(crate) const CUBE_FACES: usize = 6;
/// Total array layers: four fixtures × six faces.
pub(crate) const MAX_TORCH_LAYERS: usize = MAX_TORCH_MAPS * CUBE_FACES;
/// The byte size of [`TorchTableUniform`] — 16 + 64 + 1536; the group-3 layout's uniform size and
/// the [`SharedTorchBuffer`]'s allocation, checked against the struct below.
pub(crate) const TORCH_TABLE_BYTES: u64 = 1616;

/// **The app→render publication** (`benilla_app::torch_shadow` writes it each frame, extracted here):
/// the nearest interior fixtures' down-looking `view_proj`s + world positions and the shared interior
/// caster mesh. `count` is how many of the four slots are live this frame (0 when the lane is off).
#[derive(Resource, Clone, ExtractResource, Default)]
pub struct TorchShadowViews {
    pub count: u32,
    /// `positions[i].xyz` = fixture world position (absolute Bevy), `.w` = its range (yd).
    pub positions: [Vec4; MAX_TORCH_MAPS],
    /// The reverse-Z `view_proj` of cube face `f` of fixture `i` at `[i * CUBE_FACES + f]`
    /// (identity in unused slots).
    pub view_projs: [Mat4; MAX_TORCH_LAYERS],
    /// The interior shadow-caster mesh (retained static geometry near the camera), or `None` when the
    /// lane has not built one this frame.
    pub caster_mesh: Option<AssetId<Mesh>>,
    /// MONKEY (Phase 3B): the per-frame ENTITY caster — furniture, NPCs, the player — rebuilt every
    /// frame by the lane (entities move), drawn into every layer alongside `caster_mesh` so a table
    /// or a character throws a torch shadow on the floor too.
    pub entity_mesh: Option<AssetId<Mesh>>,
}

/// MONKEY (torch shadows Phase 3A): the ONE shared torch depth array — a `Depth32Float`
/// 512×512×24-layer `Image` asset (`RENDER_ATTACHMENT | TEXTURE_BINDING`, `RENDER_WORLD`-only,
/// no CPU data) whose sampler descriptor carries `compare: Some(GreaterEqual)`, so its `GpuImage`
/// sampler is a real comparison sampler and its default view is a `D2Array` over all 24 layers
/// (wgpu's default view dimension for a multi-layer 2D texture). Created ONCE at startup by
/// [`new_torch_shared`] (always — regardless of the interior-shadow cvars — because every model
/// material binds it; a missing image would stall every material's bind group), inserted in the
/// main world so material construction can clone the handle, and cloned into the render world so
/// the depth node and static_gx's group-3 prepare can look its `GpuImage` up.
#[derive(Resource, Clone, ExtractResource)]
pub struct TorchDepthImage(pub Handle<Image>);

/// MONKEY (torch shadows Phase 3A): the shared ≤4-fixture torch TABLE buffer — 1616 bytes of
/// [`TorchTableUniform`] (`STORAGE | COPY_DST`), the entity receiver's `torch_table` at material
/// binding 93. Created once beside [`TorchDepthImage`], cloned into every model material, and
/// rewritten in place every frame by [`upload_torch_table`] (render world, `PrepareResources`) —
/// the exact pattern of the shared light buffer (`lighting::global_light::upload_light`).
#[derive(Resource, Clone, ExtractResource)]
pub struct SharedTorchBuffer(pub Buffer);

/// The main-world pair of torch resources as the ONE system param a material-building site adds
/// beside its `SharedLightBuffer` (several of those systems sit at Bevy's 16-param ceiling).
/// [`Self::binds`] is `None` until [`new_torch_shared`] has run — the same "retry, don't bake
/// against nothing" contract the light buffer gives those sites.
#[derive(SystemParam)]
pub struct TorchShared<'w> {
    image: Option<Res<'w, TorchDepthImage>>,
    table: Option<Res<'w, SharedTorchBuffer>>,
}

impl TorchShared<'_> {
    /// The receiver bindings to clone into a `WowModelExt`, or `None` before startup created them.
    pub fn binds(&self) -> Option<TorchBinds> {
        Some(TorchBinds {
            depth: self.image.as_ref()?.0.clone(),
            table: self.table.as_ref()?.0.clone(),
        })
    }
}

/// Create the shared torch depth image + table buffer (main world, `Startup` — `RenderDevice` and
/// `Assets<Image>` are both live there). Called by the asset foundation next to the shared light
/// buffer so the pair exists before any material is built, whatever the cvars say.
pub fn new_torch_shared(
    device: &RenderDevice,
    images: &mut Assets<Image>,
) -> (TorchDepthImage, SharedTorchBuffer) {
    let image = Image {
        // A pure GPU render target: no CPU pixels, ever.
        data: None,
        texture_descriptor: TextureDescriptor {
            label: Some("torch_depth"),
            size: Extent3d {
                width: TORCH_MAP_EDGE,
                height: TORCH_MAP_EDGE,
                depth_or_array_layers: MAX_TORCH_LAYERS as u32,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Depth32Float,
            usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        },
        // The receivers' `GreaterEqual` comparison sampler (reverse-Z: lit iff the fragment's own
        // depth ≥ the stored nearest depth). `ImageSamplerDescriptor::as_wgpu` carries `compare`
        // through, so `GpuImage::sampler` IS a comparison sampler — which the `AsBindGroup`
        // derive's `sampler_type = "comparison"` binding hands to the material.
        sampler: ImageSampler::Descriptor(ImageSamplerDescriptor {
            label: Some("torch_depth_cmp".into()),
            mag_filter: ImageFilterMode::Linear,
            min_filter: ImageFilterMode::Linear,
            compare: Some(ImageCompareFunction::GreaterEqual),
            ..Default::default()
        }),
        // `None` ⇒ wgpu's default view: the texture's format, all mips/layers, and — for a D2
        // texture with >1 layers — `D2Array`, exactly the `texture_depth_2d_array` both receivers
        // declare. (Spelled out rather than relied on silently: it is the one fact the whole
        // sharing scheme rests on.)
        texture_view_descriptor: None,
        asset_usage: RenderAssetUsages::RENDER_WORLD,
        ..Default::default()
    };
    let buffer = device.create_buffer(&BufferDescriptor {
        label: Some("wow_shared_torch_table"),
        size: TORCH_TABLE_BYTES,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    (TorchDepthImage(images.add(image)), SharedTorchBuffer(buffer))
}

/// The std140/std430 `TorchTable` — mirrored in `static_gx.wgsl` (group 3, binding 2, uniform) and
/// `wow_model.wgsl` (material binding 93, storage). `count` pads to a vec4; `positions[i].xyz` is
/// the fixture, `.w` its range; `view_projs[i*6 + f]` is fixture `i`'s reverse-Z matrix for cube
/// face `f` (Phase 5: six 90° faces cover every direction, replacing the single aimed cone whose
/// edge kept eating shadows). Unused slots stay zero (never sampled: the shaders loop `count`).
/// 16 + 64 + 1536 = 1616 bytes — [`TORCH_TABLE_BYTES`]; identical bytes under both layouts
/// because every member is 16-byte aligned.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct TorchTableUniform {
    count: u32,
    _pad: [u32; 3],
    positions: [[f32; 4]; MAX_TORCH_MAPS],
    view_projs: [[[f32; 4]; 4]; MAX_TORCH_LAYERS],
}

const _: () = assert!(std::mem::size_of::<TorchTableUniform>() as u64 == TORCH_TABLE_BYTES);

impl TorchTableUniform {
    /// Pack this frame's table from the extracted views — a zero table (count 0) when the lane is
    /// off or the resource is absent, so both receivers read "no torches" rather than stale data.
    pub(crate) fn pack(views: Option<&TorchShadowViews>) -> Self {
        let mut table = Self::zeroed();
        let Some(views) = views else {
            return table;
        };
        let count = (views.count as usize).min(MAX_TORCH_MAPS);
        table.count = count as u32;
        for i in 0..count {
            table.positions[i] = views.positions[i].to_array();
            // Six cube faces per fixture: layer = fixture * 6 + face (the depth node renders and
            // the shaders' `torch_face` pick by the same index).
            for f in 0..CUBE_FACES {
                table.view_projs[i * CUBE_FACES + f] =
                    views.view_projs[i * CUBE_FACES + f].to_cols_array_2d();
            }
        }
        table
    }
}

/// Render-world: rewrite the shared torch table in place every frame, before any draw reads it
/// (`RenderSystems::PrepareResources` — the same stage `upload_light` writes the light blob). One
/// 1.6 KB upload per frame, independent of material count. static_gx's group-3 uniform is packed
/// from the same [`TorchTableUniform::pack`] in `render.rs`, so the two receivers can never
/// disagree on a fixture.
fn upload_torch_table(
    queue: Res<RenderQueue>,
    buffer: Option<Res<SharedTorchBuffer>>,
    views: Option<Res<TorchShadowViews>>,
) {
    let Some(buffer) = buffer else {
        return;
    };
    let table = TorchTableUniform::pack(views.as_deref());
    queue.write_buffer(&buffer.0, 0, bytemuck::bytes_of(&table));
}

/// The persistent render-world state that does NOT depend on the image: the comparison sampler
/// static_gx's group 3 binds (kept as its own `GreaterEqual` sampler — unchanged behaviour; the
/// material binds the image's, which is built from the same descriptor), the depth-only pipeline
/// and its per-fixture view-uniform layout. Built once in [`RenderStartup`].
#[derive(Resource)]
pub(crate) struct TorchDepthTargets {
    sampler: Sampler,
    pipeline: CachedRenderPipelineId,
    view_layout: BindGroupLayoutDescriptor,
}

impl TorchDepthTargets {
    /// The `GreaterEqual` comparison sampler for the static_gx group-3 shadow lookup.
    pub(crate) fn sampler(&self) -> &Sampler {
        &self.sampler
    }
}

/// The per-layer D2 render-attachment views over the shared image's texture, cached by that
/// texture's id: rebuilt only if the `GpuImage` is ever re-prepared (it is created once and never
/// modified, so in practice exactly once), never per frame. Empty until the image is resident.
#[derive(Resource, Default)]
struct TorchLayerViews {
    texture: Option<TextureId>,
    /// `views[i]` = the depth attachment for array layer `i` (fixture `i / 6`, face `i % 6`).
    views: Vec<TextureView>,
}

/// This frame's per-fixture view-uniform bind groups (one per live map), rebuilt in
/// [`RenderSystems::PrepareBindGroups`].
#[derive(Resource, Default)]
struct TorchDepthDraw {
    count: usize,
    /// `bind_groups[i]` binds layer `i`'s `view_proj` at group 0 of the depth pipeline.
    bind_groups: Vec<BindGroup>,
}

fn init_torch_depth(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    pipeline_cache: Res<PipelineCache>,
    asset_server: Res<AssetServer>,
) {
    let sampler = render_device.create_sampler(&SamplerDescriptor {
        label: Some("torch_depth_cmp"),
        min_filter: FilterMode::Linear,
        mag_filter: FilterMode::Linear,
        compare: Some(CompareFunction::GreaterEqual),
        ..Default::default()
    });
    // Group 0: the single per-fixture `view_proj` (a mat4, 64 bytes), vertex-only.
    let view_layout = BindGroupLayoutDescriptor::new(
        "torch_depth_view_layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::VERTEX,
            (uniform_buffer_sized(false, Some(std::num::NonZero::new(64).unwrap())),),
        ),
    );
    let shader: Handle<Shader> =
        asset_server.load("embedded://benilla_world/shaders/torch_depth.wgsl");
    let pipeline = pipeline_cache.queue_render_pipeline(RenderPipelineDescriptor {
        label: Some("torch_depth".into()),
        layout: vec![view_layout.clone()],
        vertex: VertexState {
            shader,
            shader_defs: vec![],
            entry_point: Some("vertex".into()),
            // The caster mesh (`shadow_core::empty_shadow_mesh`) carries ONLY position — stride 12.
            buffers: vec![VertexBufferLayout {
                array_stride: 12,
                step_mode: VertexStepMode::Vertex,
                attributes: vec![VertexAttribute {
                    format: VertexFormat::Float32x3,
                    offset: 0,
                    shader_location: 0,
                }],
            }],
        },
        // Depth only — no fragment, no color targets.
        fragment: None,
        primitive: PrimitiveState {
            // Cull none is safest for the mixed-winding WMO/static caster geometry.
            cull_mode: None,
            ..Default::default()
        },
        depth_stencil: Some(DepthStencilState {
            format: TextureFormat::Depth32Float,
            depth_write_enabled: true,
            depth_compare: CompareFunction::GreaterEqual,
            stencil: StencilState::default(),
            bias: DepthBiasState::default(),
        }),
        multisample: MultisampleState {
            count: 1,
            ..Default::default()
        },
        ..default()
    });
    commands.insert_resource(TorchDepthTargets {
        sampler,
        pipeline,
        view_layout,
    });
}

/// Build one view-uniform bind group per live layer from the extracted [`TorchShadowViews`], and
/// (re)build the per-layer attachment views whenever the shared image's texture changes identity.
fn prepare_torch_depth(
    mut commands: Commands,
    targets: Option<Res<TorchDepthTargets>>,
    views: Option<Res<TorchShadowViews>>,
    image: Option<Res<TorchDepthImage>>,
    images: Res<RenderAssets<GpuImage>>,
    mut layers: ResMut<TorchLayerViews>,
    pipeline_cache: Res<PipelineCache>,
    render_device: Res<RenderDevice>,
) {
    // The attachment views, cached by texture id — 24 views once, not per frame.
    let gpu_image = image.as_ref().and_then(|h| images.get(&h.0));
    match gpu_image {
        Some(gpu) if layers.texture != Some(gpu.texture.id()) => {
            layers.views = (0..MAX_TORCH_LAYERS as u32)
                .map(|i| {
                    gpu.texture.create_view(&TextureViewDescriptor {
                        label: Some("torch_depth_layer"),
                        format: Some(TextureFormat::Depth32Float),
                        dimension: Some(TextureViewDimension::D2),
                        aspect: TextureAspect::DepthOnly,
                        base_mip_level: 0,
                        mip_level_count: Some(1),
                        base_array_layer: i,
                        array_layer_count: Some(1),
                        ..default()
                    })
                })
                .collect();
            layers.texture = Some(gpu.texture.id());
        }
        Some(_) => {}
        None => {
            // Not resident (yet): nothing to render into this frame.
            layers.views.clear();
            layers.texture = None;
        }
    }
    let Some(targets) = targets else {
        commands.insert_resource(TorchDepthDraw::default());
        return;
    };
    let count = views
        .as_ref()
        .map_or(0, |v| (v.count as usize).min(MAX_TORCH_MAPS));
    if count == 0 {
        commands.insert_resource(TorchDepthDraw::default());
        return;
    }
    let views = views.expect("count > 0 implies views present");
    let layout = pipeline_cache.get_bind_group_layout(&targets.view_layout);
    // Six cube faces per live fixture: layer = fixture * CUBE_FACES + face.
    let layer_count = count * CUBE_FACES;
    let mut bind_groups = Vec::with_capacity(layer_count);
    for i in 0..layer_count {
        let vp = views.view_projs[i].to_cols_array();
        let buffer = render_device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("torch_depth_view"),
            contents: bytemuck::cast_slice(&vp),
            usage: BufferUsages::UNIFORM,
        });
        // The wgpu bind group keeps the buffer alive; the local handle can drop.
        bind_groups.push(render_device.create_bind_group(
            "torch_depth_view",
            &layout,
            &BindGroupEntries::sequential((buffer.as_entire_binding(),)),
        ));
    }
    commands.insert_resource(TorchDepthDraw {
        count: layer_count,
        bind_groups,
    });
}

#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
struct TorchDepthLabel;

#[derive(Default)]
struct TorchDepthNode;

impl ViewNode for TorchDepthNode {
    // Run only for the world view — the same marker the retained pass gates on, so a portrait-booth
    // bake never renders torch depth with the booth's matrices.
    type ViewQuery = &'static StaticGxView;

    fn run<'w>(
        &self,
        _graph: &mut RenderGraphContext,
        render_context: &mut RenderContext<'w>,
        _view: QueryItem<'w, '_, Self::ViewQuery>,
        world: &'w World,
    ) -> Result<(), NodeRunError> {
        let Some(views) = world.get_resource::<TorchShadowViews>() else {
            return Ok(());
        };
        let Some(draw) = world.get_resource::<TorchDepthDraw>() else {
            return Ok(());
        };
        let Some(targets) = world.get_resource::<TorchDepthTargets>() else {
            return Ok(());
        };
        // The shared image's per-layer attachment views — empty while the GpuImage is not resident.
        let Some(layers) = world.get_resource::<TorchLayerViews>() else {
            return Ok(());
        };
        // `count` is LAYERS here: six cube faces per live fixture.
        let count = draw
            .count
            .min(views.count as usize * CUBE_FACES)
            .min(MAX_TORCH_LAYERS)
            .min(layers.views.len());
        if count == 0 {
            return Ok(());
        }
        let pipeline_cache = world.resource::<PipelineCache>();
        let Some(pipeline) = pipeline_cache.get_render_pipeline(targets.pipeline) else {
            return Ok(());
        };
        let meshes = world.resource::<RenderAssets<RenderMesh>>();
        let allocator = world.resource::<MeshAllocator>();
        // MONKEY (Phase 3B): BOTH casters — the drift-rebuilt static interior (walls/columns) and the
        // per-frame entity set (furniture, NPCs, the player) — resolved up front. One that is not
        // resident this frame (just built, or empty) is skipped, never fatal to the other.
        let mut draws = Vec::with_capacity(2);
        for mesh_id in [views.caster_mesh, views.entity_mesh].into_iter().flatten() {
            let Some(mesh) = meshes.get(mesh_id) else {
                continue;
            };
            let (Some(vslice), Some(islice)) = (
                allocator.mesh_vertex_slice(&mesh_id),
                allocator.mesh_index_slice(&mesh_id),
            ) else {
                continue;
            };
            let RenderMeshBufferInfo::Indexed { index_format, .. } = &mesh.buffer_info else {
                continue;
            };
            if islice.range.is_empty() {
                continue;
            }
            draws.push((vslice, islice, *index_format));
        }
        if draws.is_empty() {
            return Ok(());
        }
        for i in 0..count {
            let mut pass = render_context.begin_tracked_render_pass(RenderPassDescriptor {
                label: Some("torch_depth"),
                color_attachments: &[],
                depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
                    view: &layers.views[i],
                    // Reverse-Z: clear to 0.0 (far), keep the nearest surface (GreaterEqual).
                    depth_ops: Some(Operations {
                        load: LoadOp::Clear(0.0),
                        store: StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            pass.set_render_pipeline(pipeline);
            pass.set_bind_group(0, &draw.bind_groups[i], &[]);
            // Static walls + moving entities into the SAME layer under the one clear.
            for (vslice, islice, index_format) in &draws {
                pass.set_vertex_buffer(0, vslice.buffer.slice(..));
                pass.set_index_buffer(islice.buffer.slice(..), *index_format);
                pass.draw_indexed(
                    islice.range.start..islice.range.end,
                    i32::try_from(vslice.range.start).unwrap_or(0),
                    0..1,
                );
            }
        }
        Ok(())
    }
}

/// Wire the ALWAYS-ON half (MONKEY, Phase 3A): the extract plugins for the shared image + table
/// (so the render world can look up the `GpuImage` and write the buffer), and the per-frame table
/// upload. Registered by the asset foundation next to the shared light buffer — NOT gated on the
/// retained pass (`static_gx::enabled`), because every model material binds these whether or not
/// static_gx draws. With the depth node off the table reads count 0 and the image stays at its
/// zero-initialised (= reverse-Z far ⇒ unshadowed) contents, so entities simply receive no shadow.
pub(crate) fn register_shared(app: &mut App) {
    app.add_plugins((
        ExtractResourcePlugin::<TorchDepthImage>::default(),
        ExtractResourcePlugin::<SharedTorchBuffer>::default(),
    ));
    if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
        render_app.add_systems(
            Render,
            upload_torch_table.in_set(RenderSystems::PrepareResources),
        );
    }
}

/// Wire the render half: the persistent targets (RenderStartup), the per-frame view bind groups, and
/// the depth node between the prepass and the main opaque pass.
pub(super) fn build(app: &mut App) {
    let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
        return;
    };
    render_app
        .init_resource::<TorchLayerViews>()
        .add_systems(RenderStartup, init_torch_depth)
        .add_systems(
            Render,
            prepare_torch_depth.in_set(RenderSystems::PrepareBindGroups),
        )
        .add_render_graph_node::<ViewNodeRunner<TorchDepthNode>>(Core3d, TorchDepthLabel)
        .add_render_graph_edges(
            Core3d,
            // After all prepasses complete, before the main opaque pass — so the depth maps are
            // ready when `StaticGxNode` (between opaque and transparent) samples them.
            (
                Node3d::EndPrepasses,
                TorchDepthLabel,
                Node3d::MainOpaquePass,
            ),
        );
}
