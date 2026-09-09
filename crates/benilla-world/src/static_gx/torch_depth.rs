//! MONKEY (static torch cache): persistent six-face static maps for up to sixteen fixtures.
//! Bevy point-light clustering is disabled on the world view, so this dedicated depth lane
//! supplies both receivers. Static geometry renders only on promotion/residency/fixture changes;
//! nearest-N fixtures copy their six cached faces into a live bank and overlay moving entities.
//! With the defaults (12 resident, 4 dynamic), a steady frame records 24 copies, 24 entity draws
//! and zero static draws. At most two dirty fixtures rebuild per frame; physical slots survive
//! contribution ranking and cross-fades. See benilla_app::torch_shadow for eligibility and keys.
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
//! sampler. The 16-fixture table gets the same treatment ([`SharedTorchBuffer`]: one raw buffer,
//! rewritten in place every frame from [`TorchShadowViews`] with the SAME [`TorchTableUniform`]
//! bytes static_gx's group-3 uniform carries).
//!
//! Reverse-Z throughout (the whole engine is — `capture::depth_probe`): the depth pass clears to 0.0
//! (= far), keeps the nearest surface with `GreaterEqual`, and the comparison sampler is `GreaterEqual`
//! too. The per-fixture `view_proj` is a reverse-Z perspective built app-side in f64 (fixtures sit at
//! absolute Bevy world coordinates, ~9,300 out) and downcast to f32.

use std::hash::{Hash, Hasher};
use std::sync::Mutex;

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

/// MONKEY (static torch cache): each 512-square Depth32 face is 1 MiB. Two banks support
/// sixteen resident static cubes and up to sixteen live entity overlays: 192 MiB total. Keep
/// the edge here so a memory-constrained port can choose 384 (108 MiB, 25% less linear detail).
const TORCH_MAP_EDGE: u32 = 512;
pub(crate) const MAX_TORCH_MAPS: usize = 16;
pub(crate) const CUBE_FACES: usize = 6;
/// MONKEY (static torch cache): matrices address the 96 STATIC faces. Physical texture layers
/// 0..96 hold those faces permanently; 96..192 are live copies used only by dynamic-mask bits.
pub(crate) const MAX_TORCH_LAYERS: usize = MAX_TORCH_MAPS * CUBE_FACES;
const TORCH_TEXTURE_LAYERS: usize = MAX_TORCH_LAYERS * 2;
/// MONKEY (static torch cache): count@0 (16), positions@16 (256), view_projs@272 (6144).
/// Total 6416 bytes, identical under WGSL uniform/storage alignment. count.z is the live-bank
/// bit mask, count.w reserved. Both WGSL TorchTable copies and render.rs MUST agree with this.
pub(crate) const TORCH_TABLE_BYTES: u64 = 6416;

/// **The app→render publication** (`benilla_app::torch_shadow` writes it each frame, extracted here):
/// the promoted fixtures' cube-face matrices, world positions and static mesh revisions.
/// `count` is the sparse slot high-water mark (0 when the lane is off).
#[derive(Resource, Clone, ExtractResource)]
pub struct TorchShadowViews {
    pub count: u32,
    // MONKEY (static torch cache): sparse physical slots; count is the high-water mark, not the
    // population. Holes have weight zero. ready_mask is assigned only in render preparation.
    pub dynamic_mask: u32,
    pub ready_mask: u32,
    /// MONKEY (torch caster selection): the live PCF tap-radius scale (`interiorShadowSoft`,
    /// 0.5..3). Carried here rather than in `DynamicInteriors` because it belongs to the shadow
    /// table's own bytes — [`TorchTableUniform::pack`] puts it in the otherwise-padding `count.y`.
    pub soft: f32,
    /// `positions[i].xyz` = fixture world position (absolute Bevy), `.w` = the slot's FADE WEIGHT
    /// `0..1` (MONKEY, torch caster selection). The `.w` lane used to carry the fixture's range,
    /// which no shader ever read; it now carries the cross-fade, and both receivers return
    /// `mix(1.0, shadow, w)` so a promoted fixture's shadow ramps in over ~1/3 s and a demoted one
    /// ramps out before its slot is reused — the fix for the pop this whole change is about.
    pub positions: [Vec4; MAX_TORCH_MAPS],
    /// The reverse-Z `view_proj` of cube face `f` of fixture `i` at `[i * CUBE_FACES + f]`
    /// (identity in unused slots).
    pub view_projs: [Mat4; MAX_TORCH_LAYERS],
    /// MONKEY (static torch cache): fresh asset identity per geometry revision/slot owner;
    /// None while the app's two-rebuild budget defers this slot. Static uploads must be resident
    /// before depth can be marked valid. Empty resident meshes still clear all six faces.
    pub caster_meshes: [Option<AssetId<Mesh>>; MAX_TORCH_MAPS],
    /// MONKEY (static torch cache): moving/skinned entities only, regenerated each frame.
    pub entity_mesh: Option<AssetId<Mesh>>,
}

/// Hand-written because `[Mat4; 96]` has no `Default` — std only derives array `Default` up to
/// `N = 32`, so the derive that worked at 24 layers stops compiling at 96. (It also lets `soft`
/// default to a sane `1.0` rather than a zero tap radius.)
impl Default for TorchShadowViews {
    fn default() -> Self {
        Self {
            count: 0,
            dynamic_mask: 0,
            ready_mask: 0,
            soft: 1.0,
            positions: [Vec4::ZERO; MAX_TORCH_MAPS],
            view_projs: [Mat4::ZERO; MAX_TORCH_LAYERS],
            caster_meshes: [None; MAX_TORCH_MAPS],
            entity_mesh: None,
        }
    }
}

/// MONKEY (torch shadows Phase 3A): the ONE shared torch depth array — a `Depth32Float`
/// MONKEY (static torch cache): 512×512×192-layer `Image` asset (render, sample, copy-src/dst),
/// no CPU data) whose sampler descriptor carries `compare: Some(GreaterEqual)`, so its `GpuImage`
/// sampler is a real comparison sampler and its default view is a `D2Array` over all 192 layers
/// (wgpu's default view dimension for a multi-layer 2D texture). Created ONCE at startup by
/// [`new_torch_shared`] (always — regardless of the interior-shadow cvars — because every model
/// material binds it; a missing image would stall every material's bind group), inserted in the
/// main world so material construction can clone the handle, and cloned into the render world so
/// the depth node and static_gx's group-3 prepare can look its `GpuImage` up.
#[derive(Resource, Clone, ExtractResource)]
pub struct TorchDepthImage(pub Handle<Image>);

/// MONKEY (torch shadows Phase 3A): the shared 16-fixture torch TABLE buffer — 6416 bytes of
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
                depth_or_array_layers: TORCH_TEXTURE_LAYERS as u32,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Depth32Float,
            // MONKEY (static torch cache): wgpu 27 transfer.rs permits partial ARRAY layers
            // for depth, provided x/y cover the entire mip. Banks are disjoint subresources.
            usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING
                | TextureUsages::COPY_SRC | TextureUsages::COPY_DST,
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

/// MONKEY (static torch cache): byte-identical std140/std430 table in BOTH receiver shaders.
/// count@0: live high-water mark, soft*100, dynamic mask, reserved (16 bytes).
/// positions[16]@16: world xyz + fade weight (256 bytes).
/// view_projs[96]@272: six static-bank face matrices per physical slot (6144 bytes).
/// Total 6416. Unready/hole slots have zero weight; no stale owner's depth can be sampled.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct TorchTableUniform {
    count: u32,
    /// MONKEY (torch caster selection): `soft x 100` — the PCF tap-radius scale, riding the first
    /// of the three padding words the `vec4<u32>` alignment already forced us to carry. The
    /// alternative was a whole new 16-byte row for one dial.
    soft_x100: u32,
    dynamic_mask: u32,
    _pad: u32,
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
        table.dynamic_mask = views.dynamic_mask & views.ready_mask;
        // MONKEY (torch caster selection): 0 would mean a zero tap radius (four identical taps =
        // a hard edge), so a table published without a scale reads as the neutral 1.0.
        let soft = if views.soft > 0.01 { views.soft } else { 1.0 };
        table.soft_x100 = (soft * 100.0).round().max(1.0) as u32;
        for i in 0..count {
            table.positions[i] = views.positions[i].to_array();
            if views.ready_mask & (1 << i) == 0 { table.positions[i][3] = 0.0; }
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
/// MONKEY (static torch cache): 6416-byte upload per frame, independent of material count. static_gx's group-3 uniform is packed
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

/// MONKEY (static torch cache): only a completed six-face command recording certifies a cache
/// entry. Preparation may run while there is no world view or the pipeline is still compiling;
/// such a frame must not mark a never-rendered map cached. Texture recreation resets all keys.
#[derive(Resource, Default)]
struct TorchStaticCache(Mutex<[Option<AssetId<Mesh>>; MAX_TORCH_MAPS]>);

// MONKEY (static torch cache): the app logs requests; this records actual GPU cache work.
#[derive(Resource, Default)]
struct TorchCacheTrace(Mutex<Option<std::time::Instant>>);

/// This frame's per-fixture view-uniform bind groups (one per live map), rebuilt in
/// MONKEY (static torch cache): retained across frames; refreshed in PrepareResources.
#[derive(Resource, Default)]
struct TorchDepthDraw {
    view_projs: Vec<Mat4>,
    count: usize,
    rebuild_mask: u32,
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
    mut draw: ResMut<TorchDepthDraw>,
    targets: Option<Res<TorchDepthTargets>>,
    mut views: Option<ResMut<TorchShadowViews>>,
    cache: Res<TorchStaticCache>,
    meshes: Res<RenderAssets<RenderMesh>>,
    allocator: Res<MeshAllocator>,
    image: Option<Res<TorchDepthImage>>,
    images: Res<RenderAssets<GpuImage>>,
    mut layers: ResMut<TorchLayerViews>,
    pipeline_cache: Res<PipelineCache>,
    render_device: Res<RenderDevice>,
) {
    draw.count = 0;
    draw.rebuild_mask = 0;
    // MONKEY (static torch cache): fail closed until every prerequisite for this frame exists.
    if let Some(views) = views.as_mut() { views.ready_mask = 0; }
    // MONKEY (static torch cache): attachment views are allocated once per texture identity.
    let gpu_image = image.as_ref().and_then(|h| images.get(&h.0));
    match gpu_image {
        Some(gpu) if layers.texture != Some(gpu.texture.id()) => {
            *cache.0.lock().unwrap() = [None; MAX_TORCH_MAPS];
            layers.views = (0..TORCH_TEXTURE_LAYERS as u32)
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
        return;
    };
    let count = views
        .as_ref()
        .map_or(0, |v| (v.count as usize).min(MAX_TORCH_MAPS));
    if count == 0 {
        return;
    }
    let mut views = views.expect("count > 0 implies views present");
    if layers.texture.is_none() || pipeline_cache.get_render_pipeline(targets.pipeline).is_none() {
        return;
    }
    let cached = cache.0.lock().unwrap();
    let (ready_mask, rebuild_mask) = torch_cache_plan(&*cached, &views.caster_meshes[..count],
        |id| torch_mesh_ready(id, &meshes, &allocator));
    views.ready_mask = ready_mask;
    // If entity upload is late, bind the static bank for this frame, never last frame's overlay.
    if !views.entity_mesh.is_some_and(|id| torch_mesh_ready(id, &meshes, &allocator)) {
        views.dynamic_mask = 0;
    }
    let layout = pipeline_cache.get_bind_group_layout(&targets.view_layout);
    // Six cube faces per live fixture: layer = fixture * CUBE_FACES + face.
    let layer_count = count * CUBE_FACES;
    for i in 0..layer_count {
        // MONKEY (static torch cache): fixture projections do not move with the camera.
        // Keep their buffers/bind groups too: zero resource allocation on a steady frame.
        if draw.view_projs.get(i) == Some(&views.view_projs[i]) { continue; }
        let vp = views.view_projs[i].to_cols_array();
        let buffer = render_device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("torch_depth_view"),
            contents: bytemuck::cast_slice(&vp),
            usage: BufferUsages::UNIFORM,
        });
        // The wgpu bind group keeps the buffer alive; the local handle can drop.
        let bind_group = render_device.create_bind_group(
            "torch_depth_view",
            &layout,
            &BindGroupEntries::sequential((buffer.as_entire_binding(),)),
        );
        if i < draw.bind_groups.len() {
            draw.bind_groups[i] = bind_group;
            draw.view_projs[i] = views.view_projs[i];
        } else {
            draw.bind_groups.push(bind_group);
            draw.view_projs.push(views.view_projs[i]);
        }
    }
    draw.count = layer_count;
    draw.rebuild_mask = rebuild_mask;
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
        let image = world.resource::<TorchDepthImage>();
        let images = world.resource::<RenderAssets<GpuImage>>();
        let Some(gpu) = images.get(&image.0) else { return Ok(()) };
        let cache = world.resource::<TorchStaticCache>();
        let mut cached = cache.0.lock().unwrap();
        let trace = world.resource::<TorchCacheTrace>();
        let trace_on = std::env::var_os("WOW_TORCH_TRACE").is_some();
        let mut last_trace = trace.0.lock().unwrap();
        let trace_due = trace_on && last_trace.is_none_or(|t| t.elapsed().as_secs_f32() >= 1.0);
        if trace_due { *last_trace = Some(std::time::Instant::now()); }
        for slot in 0..count / CUBE_FACES {
            if views.ready_mask & (1 << slot) == 0 { continue; }
            let rebuild = draw.rebuild_mask & (1 << slot) != 0
                && cached[slot] != views.caster_meshes[slot];
            let dynamic = views.dynamic_mask & (1 << slot) != 0;
            // MONKEY (static torch cache): render static once, then copy to the disjoint live
            // bank BEFORE each entity pass. GreaterEqual + Load preserves the nearest surface.
            for face in 0..CUBE_FACES {
                let layer = slot * CUBE_FACES + face;
                if rebuild {
                    let mut pass = render_context.begin_tracked_render_pass(RenderPassDescriptor {
                        label: Some("torch_static_rebuild"),
                        color_attachments: &[],
                        depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
                            view: &layers.views[layer],
                            depth_ops: Some(Operations { load: LoadOp::Clear(0.0), store: StoreOp::Store }),
                            stencil_ops: None,
                        }),
                        timestamp_writes: None,
                        occlusion_query_set: None,
                    });
                    pass.set_render_pipeline(pipeline);
                    pass.set_bind_group(0, &draw.bind_groups[layer], &[]);
                    draw_torch_mesh(&mut pass, views.caster_meshes[slot], meshes, allocator);
                }
                if dynamic {
                    let live = layer + MAX_TORCH_LAYERS;
                    render_context.command_encoder().copy_texture_to_texture(
                        TexelCopyTextureInfo {
                            texture: &gpu.texture, mip_level: 0,
                            origin: Origin3d { x: 0, y: 0, z: layer as u32 },
                            aspect: TextureAspect::DepthOnly,
                        },
                        TexelCopyTextureInfo {
                            texture: &gpu.texture, mip_level: 0,
                            origin: Origin3d { x: 0, y: 0, z: live as u32 },
                            aspect: TextureAspect::DepthOnly,
                        },
                        Extent3d { width: TORCH_MAP_EDGE, height: TORCH_MAP_EDGE, depth_or_array_layers: 1 },
                    );
                    let mut pass = render_context.begin_tracked_render_pass(RenderPassDescriptor {
                        label: Some("torch_dynamic_overlay"),
                        color_attachments: &[],
                        depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
                            view: &layers.views[live],
                            depth_ops: Some(Operations { load: LoadOp::Load, store: StoreOp::Store }),
                            stencil_ops: None,
                        }),
                        timestamp_writes: None,
                        occlusion_query_set: None,
                    });
                    pass.set_render_pipeline(pipeline);
                    pass.set_bind_group(0, &draw.bind_groups[layer], &[]);
                    draw_torch_mesh(&mut pass, views.entity_mesh, meshes, allocator);
                }
            }
            if rebuild { cached[slot] = views.caster_meshes[slot]; }
            if trace_on && (trace_due || rebuild) {
                info!("torch-cache: slot {slot} static {} dynamic {}",
                    if rebuild { "rebuilt" } else { "cached" },
                    if dynamic { "yes" } else { "no" });
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
        .init_resource::<TorchStaticCache>()
        .init_resource::<TorchDepthDraw>()
        .init_resource::<TorchCacheTrace>()
        .add_systems(RenderStartup, init_torch_depth)
        .add_systems(
            Render,
            // MONKEY (static torch cache): ready/dynamic masks must be final before BOTH
            // the shared storage upload and static_gx's PrepareBindGroups uniform pack.
            prepare_torch_depth.in_set(RenderSystems::PrepareResources).before(upload_torch_table),
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

// MONKEY (static torch cache): planning does not commit residency. A missing upload cannot
// certify an empty map, and the two-rebuild budget also covers GPU texture recreation.
fn torch_cache_plan<T: Copy + Eq>(cached: &[Option<T>], requested: &[Option<T>],
    mut mesh_ready: impl FnMut(T) -> bool) -> (u32, u32) {
    let (mut ready, mut rebuild) = (0u32, 0u32);
    for (i, requested) in requested.iter().take(MAX_TORCH_MAPS).enumerate() {
        let Some(id) = *requested else { continue };
        if cached[i] == Some(id) {
            ready |= 1 << i;
        } else if rebuild.count_ones() < 2 && mesh_ready(id) {
            rebuild |= 1 << i;
            ready |= 1 << i;
        }
    }
    (ready, rebuild)
}

// MONKEY (static torch cache): an empty mesh is a valid all-clear cube. A nonempty mesh without
// allocator slices is still uploading, and MUST NOT become a permanently cached empty shadow.
fn torch_mesh_ready(id: AssetId<Mesh>, meshes: &RenderAssets<RenderMesh>, allocator: &MeshAllocator) -> bool {
    let Some(mesh) = meshes.get(id) else { return false };
    match mesh.buffer_info {
        RenderMeshBufferInfo::Indexed { count, .. } => count == 0
            || (allocator.mesh_vertex_slice(&id).is_some() && allocator.mesh_index_slice(&id).is_some()),
        _ => false,
    }
}

fn draw_torch_mesh<'a>(
    pass: &mut bevy::render::render_phase::TrackedRenderPass<'a>, id: Option<AssetId<Mesh>>,
    meshes: &'a RenderAssets<RenderMesh>, allocator: &'a MeshAllocator,
) {
    let Some(id) = id else { return };
    let Some(mesh) = meshes.get(id) else { return };
    let RenderMeshBufferInfo::Indexed { index_format, count } = mesh.buffer_info else { return };
    if count == 0 { return; }
    let (Some(v), Some(i)) = (allocator.mesh_vertex_slice(&id), allocator.mesh_index_slice(&id)) else { return };
    pass.set_vertex_buffer(0, v.buffer.slice(..));
    pass.set_index_buffer(i.buffer.slice(..), index_format);
    pass.draw_indexed(i.range.clone(), i32::try_from(v.range.start).unwrap_or(0), 0..1);
}

impl super::StaticGx {
    /// MONKEY (static torch cache): fixture-local static collection uses batch bounds, not the
    /// building's placement origin. A distant Abbey room is still part of a WMO anchored more
    /// than 48 yards away. The key and the mesh MUST use this identical admission predicate.
    pub fn append_torch_triangles(&self, center: Vec3, reach: f32,
        positions: &mut Vec<[f32; 3]>, indices: &mut Vec<u32>) {
        for cell in self.cells.values().chain(self.wmos.values()).chain(self.props.values()) {
            for item in &cell.items {
                if !torch_item_in_range(item, center, reach) { continue; }
                let base = positions.len() as u32;
                positions.extend(item.geometry.positions.iter().map(|p|
                    item.transform.transform_point(benilla_assets::coords::wow_to_bevy(*p)).to_array()));
                let added = positions.len() as u32 - base;
                for tri in item.geometry.indices.chunks_exact(3) {
                    if tri.iter().all(|i| *i < added) { indices.extend(tri.iter().map(|i| base + *i)); }
                }
            }
        }
    }

    /// MONKEY (static torch cache): residency fingerprint of the exact opaque source set used
    /// by append_torch_triangles. Order-independent per-item hashing ignores HashMap reordering
    /// and camera/portal/fader bookkeeping. Arrival, unload, replacement or transform edits in
    /// THIS fixture's sphere invalidate it even when neither camera nor fixture moves.
    pub fn torch_geometry_key(&self, center: Vec3, reach: f32) -> u64 {
        let mut sum = 0u64;
        for cell in self.cells.values().chain(self.wmos.values()).chain(self.props.values()) {
            for item in &cell.items {
                if !torch_item_in_range(item, center, reach) { continue; }
                let mut hash = std::collections::hash_map::DefaultHasher::new();
                // MONKEY (static torch cache): source residency stamp prevents allocator address
                // reuse after unload/reload from impersonating the previous geometry Arc.
                cell.last_change.hash(&mut hash);
                (std::sync::Arc::as_ptr(&item.geometry) as usize).hash(&mut hash);
                item.transform.to_matrix().to_cols_array().map(f32::to_bits).hash(&mut hash);
                sum = sum.wrapping_add(hash.finish());
            }
        }
        sum
    }
}

// MONKEY (static torch cache): a conservative transformed bounding sphere, independent of
// portal visibility. Missing bounds admit the source rather than silently losing a wall.
fn torch_item_in_range(item: &super::GxItem, center: Vec3, reach: f32) -> bool {
    !item.cutout && item.local_aabb.is_none_or(|aabb| torch_bound_in_range(
        &item.transform, aabb.center.into(), aabb.half_extents.into(), center, reach))
}

fn torch_bound_in_range(transform: &Transform, local_center: Vec3, half_extents: Vec3,
    center: Vec3, reach: f32) -> bool {
    let origin = transform.transform_point(local_center);
    let radius = (half_extents * transform.scale.abs()).length();
    origin.distance_squared(center) <= (reach + radius) * (reach + radius)
}

#[cfg(test)]
mod tests {
    use super::*;

    // MONKEY (static torch cache): unchanged owners do no work; replacement, streaming,
    // absent uploads, sparse holes and texture recreation cannot certify stale depth.
    #[test]
    fn cache_plan_reuses_and_limits_rebuilds() {
        let cached = [Some(10), Some(11), None, Some(13)];
        assert_eq!(torch_cache_plan(&cached, &cached, |_| false), (0b1011, 0));
        let requested = [Some(20), Some(11), Some(12), Some(23)];
        assert_eq!(torch_cache_plan(&cached, &requested, |_| true), (0b0111, 0b0101));
        assert_eq!(torch_cache_plan(&cached, &requested, |id| id == 23), (0b1010, 0b1000));
        assert_eq!(torch_cache_plan(&cached, &[None, Some(11)], |_| true), (0b10, 0));
        assert_eq!(torch_cache_plan(&[None; 4], &requested, |_| true), (0b11, 0b11));
    }

    #[test]
    fn torch_bounds_include_remote_building_batches() {
        let transform = Transform::from_xyz(200.0, 0.0, 0.0).with_scale(Vec3::splat(2.0));
        assert!(torch_bound_in_range(&transform, Vec3::new(-100.0, 0.0, 0.0),
            Vec3::splat(10.0), Vec3::ZERO, 48.0));
        assert!(!torch_bound_in_range(&transform, Vec3::ZERO,
            Vec3::splat(10.0), Vec3::ZERO, 48.0));
    }

    // MONKEY (static torch cache): protect uniform/storage byte agreement and sparse-slot bank
    // selection, including the highest slot and an upload that has not reached the GPU yet.
    #[test]
    fn torch_table_layout_and_readiness() {
        assert_eq!(std::mem::size_of::<TorchTableUniform>(), 6416);
        assert_eq!(std::mem::offset_of!(TorchTableUniform, positions), 16);
        assert_eq!(std::mem::offset_of!(TorchTableUniform, view_projs), 272);
        let mut views = TorchShadowViews { count: 16, dynamic_mask: (1 << 15) | 1,
            ready_mask: 1 << 15, ..Default::default() };
        views.positions[0] = Vec4::ONE;
        views.positions[15] = Vec4::ONE;
        let table = TorchTableUniform::pack(Some(&views));
        assert_eq!(table.dynamic_mask, 1 << 15);
        assert_eq!(table.positions[0][3], 0.0);
        assert_eq!(table.positions[15][3], 1.0);
        assert_eq!(MAX_TORCH_LAYERS, 96);
        assert_eq!(TORCH_TEXTURE_LAYERS, 192);
    }
}
