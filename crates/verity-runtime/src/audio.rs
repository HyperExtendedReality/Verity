//! Audio Source components, played through the output device.

use std::{collections::HashMap, sync::Arc};

use serde::{Deserialize, Serialize};
use verity_audio::{Audio, Clip, Falloff, Listener, Sound, VoiceId};
use verity_core::{Component, EntityHandle, FieldMut, World};

/// Plays a sound file from its entity when the scene plays.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct AudioSourceComponent {
    /// A WAV file.
    pub clip: String,
    pub volume: f32,
    pub looping: bool,
    pub play_on_start: bool,
    /// Heard from the entity, quieter with distance; otherwise heard
    /// everywhere alike.
    pub spatial: bool,
    /// Full volume within this distance.
    pub near: f32,
    /// Silent beyond this distance.
    pub far: f32,
}

impl Default for AudioSourceComponent {
    fn default() -> Self {
        Self {
            clip: String::new(),
            volume: 1.0,
            looping: false,
            play_on_start: true,
            spatial: true,
            near: 1.0,
            far: 50.0,
        }
    }
}

impl Component for AudioSourceComponent {
    const NAME: &'static str = "Audio Source";

    fn fields(&mut self) -> Vec<(&'static str, FieldMut<'_>)> {
        vec![
            ("Clip", FieldMut::File(&mut self.clip, &["wav"])),
            ("Volume", FieldMut::Float(&mut self.volume, 0.0..=4.0)),
            ("Loop", FieldMut::Bool(&mut self.looping)),
            ("Play on start", FieldMut::Bool(&mut self.play_on_start)),
            ("Spatial", FieldMut::Bool(&mut self.spatial)),
            ("Near", FieldMut::Float(&mut self.near, 0.01..=1_000.0)),
            ("Far", FieldMut::Float(&mut self.far, 0.1..=100_000.0)),
        ]
    }
}

impl AudioSourceComponent {
    fn sound(&self, position: [f32; 3]) -> Sound {
        Sound {
            volume: self.volume,
            looping: self.looping,
            spatial: self.spatial.then_some(Falloff {
                near: self.near,
                far: self.far,
            }),
            position: position.into(),
        }
    }
}

/// A world's Audio Sources playing on the output device; dropping it stops them.
pub struct SceneAudio {
    output: Audio,
    /// Spatial voices, which follow their entities.
    voices: Vec<(EntityHandle, VoiceId)>,
}

impl SceneAudio {
    /// Opens the output device and starts every source that plays on start.
    /// Each file loads once, however many sources share it.
    pub fn start(world: &World) -> Result<Self, String> {
        let mut clips: HashMap<&str, Arc<Clip>> = HashMap::new();
        let mut starting = Vec::new();
        for (entity, source) in world.query::<AudioSourceComponent>() {
            if !source.play_on_start || source.clip.is_empty() {
                continue;
            }
            let name = world.get(entity).map_or("", |e| e.name);
            let clip = match clips.get(source.clip.as_str()) {
                Some(clip) => clip.clone(),
                None => {
                    let error = |e: String| format!("{name}: Audio Source: {}: {e}", source.clip);
                    let bytes = std::fs::read(&source.clip).map_err(|e| error(e.to_string()))?;
                    let clip = Arc::new(Clip::from_wav(&bytes).map_err(error)?);
                    clips.insert(&source.clip, clip.clone());
                    clip
                }
            };
            starting.push((entity, clip, source));
        }
        let mut output = Audio::open()?;
        let mut voices = Vec::new();
        for (entity, clip, source) in starting {
            let at = world.get(entity).expect("queried from this world").transform;
            let voice = output.play(clip, source.sound(at.translation));
            if source.spatial {
                voices.push((entity, voice));
            }
        }
        Ok(Self { output, voices })
    }

    /// Hear the world from `listener`, with sources where their entities are.
    pub fn sync(&self, world: &World, listener: Listener) {
        self.output.set_listener(listener);
        for &(entity, voice) in &self.voices {
            if let Some(entity) = world.get(entity) {
                self.output
                    .set_position(voice, entity.transform.translation.into());
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use verity_core::Scene;

    #[test]
    fn a_missing_clip_is_reported_by_entity_before_any_device_opens() {
        let mut scene = Scene::default();
        let source = AudioSourceComponent {
            clip: "no/such/file.wav".into(),
            ..AudioSourceComponent::default()
        };
        (scene.entities[0].components)
            .insert("Audio Source".into(), serde_json::to_value(source).unwrap());
        let world = World::from_scene(&scene, &crate::registry()).unwrap();
        let error = SceneAudio::start(&world).err().unwrap();
        assert!(error.starts_with(&format!("{}: Audio Source", scene.entities[0].name)), "{error}");
    }
}
