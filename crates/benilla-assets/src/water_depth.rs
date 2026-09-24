//! Shared water settings and image binding, independent of world streaming.
use bevy::prelude::*;

/// 0 = Classic, 1 = Enhanced, 2 = High (currently the same stage-1 effects).
#[derive(Resource, Clone, Copy, PartialEq)]
pub struct WaterQuality(pub u8);

impl Default for WaterQuality {
    fn default() -> Self { Self(1) }
}

/// The world view's opaque depth, resolved to a sampleable R32Float image.
#[derive(Resource, Clone)]
pub struct WaterDepthImage(pub Handle<Image>);

impl FromWorld for WaterDepthImage {
    fn from_world(world: &mut World) -> Self {
        use bevy::{asset::RenderAssetUsages, render::render_resource::*};
        let mut image = Image::new_fill(
            Extent3d::default(), TextureDimension::D2, &0f32.to_le_bytes(),
            TextureFormat::R32Float, RenderAssetUsages::default(),
        );
        image.texture_descriptor.usage = TextureUsages::TEXTURE_BINDING
            | TextureUsages::RENDER_ATTACHMENT | TextureUsages::COPY_DST;
        Self(world.resource_mut::<Assets<Image>>().add(image))
    }
}
