//! Mixing playing sounds into an output buffer.

use std::{f32::consts::FRAC_PI_4, sync::Arc};

use glam::Vec3;

use crate::Clip;

/// Identifies one playing sound.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct VoiceId(pub(crate) u64);

/// How a sound plays.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sound {
    pub volume: f32,
    pub looping: bool,
    /// Heard from `position`, or `None` for sound that plays straight to the
    /// speakers, like music.
    pub spatial: Option<Falloff>,
    pub position: Vec3,
}

impl Default for Sound {
    fn default() -> Self {
        Self {
            volume: 1.0,
            looping: false,
            spatial: None,
            position: Vec3::ZERO,
        }
    }
}

/// Full volume within `near`, fading with distance to silence at `far`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Falloff {
    pub near: f32,
    pub far: f32,
}

impl Falloff {
    pub fn gain(self, distance: f32) -> f32 {
        let near = self.near.max(1e-3);
        let far = self.far.max(near);
        let fade = (1.0 - (distance - near) / (far - near).max(1e-3)).clamp(0.0, 1.0);
        near / distance.max(near) * fade
    }
}

/// Where sound is heard from.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Listener {
    pub position: Vec3,
    /// The listener's right, a unit vector.
    pub right: Vec3,
}

impl Default for Listener {
    fn default() -> Self {
        Self {
            position: Vec3::ZERO,
            right: Vec3::X,
        }
    }
}

/// A change for the mixer, sent to the audio thread.
#[derive(Debug)]
pub enum Command {
    Play(VoiceId, Arc<Clip>, Sound),
    Move(VoiceId, Vec3),
    Stop(VoiceId),
    Listener(Listener),
}

struct Voice {
    id: VoiceId,
    clip: Arc<Clip>,
    sound: Sound,
    /// Position in clip frames.
    at: f64,
}

/// Plays sounds into stereo output.
#[derive(Default)]
pub struct Mixer {
    voices: Vec<Voice>,
    listener: Listener,
}

impl Mixer {
    pub fn apply(&mut self, command: Command) {
        match command {
            Command::Play(id, clip, sound) if clip.frames() > 0 => self.voices.push(Voice {
                id,
                clip,
                sound,
                at: 0.0,
            }),
            Command::Play(..) => {}
            Command::Move(id, position) => {
                if let Some(voice) = self.voices.iter_mut().find(|v| v.id == id) {
                    voice.sound.position = position;
                }
            }
            Command::Stop(id) => self.voices.retain(|v| v.id != id),
            Command::Listener(listener) => self.listener = listener,
        }
    }

    pub fn playing(&self) -> usize {
        self.voices.len()
    }

    /// Fill `out`, interleaved with `channels` per frame at `rate` Hz, with
    /// every playing sound. Channels past the second stay silent.
    pub fn render(&mut self, out: &mut [f32], channels: usize, rate: u32) {
        out.fill(0.0);
        if channels == 0 || rate == 0 {
            return;
        }
        let listener = self.listener;
        self.voices.retain_mut(|voice| {
            let clip = &*voice.clip;
            let frames = clip.frames();
            let step = clip.rate as f64 / rate as f64;
            let sound = voice.sound;
            // shortcut: gains change per buffer, not per sample, so a fast
            // mover can click; ramp them if that is ever heard.
            let (left, right) = match sound.spatial {
                None => (sound.volume, sound.volume),
                Some(falloff) => {
                    let offset = sound.position - listener.position;
                    let gain = sound.volume * falloff.gain(offset.length());
                    let side = offset.normalize_or_zero().dot(listener.right);
                    let angle = (side + 1.0) * FRAC_PI_4;
                    (gain * angle.cos(), gain * angle.sin())
                }
            };
            let stride = clip.channels as usize;
            let sample = |frame: usize, channel: usize| clip.samples[frame * stride + channel];
            for frame in out.chunks_exact_mut(channels) {
                if voice.at >= frames as f64 {
                    if !sound.looping {
                        return false;
                    }
                    voice.at %= frames as f64;
                }
                let index = voice.at as usize;
                let t = (voice.at - index as f64) as f32;
                let next = if index + 1 < frames {
                    index + 1
                } else if sound.looping {
                    0
                } else {
                    index
                };
                let read = |c: usize| sample(index, c) + (sample(next, c) - sample(index, c)) * t;
                let (l, r) = if stride == 1 {
                    (read(0), read(0))
                } else if sound.spatial.is_some() {
                    let mono = (read(0) + read(1)) * 0.5;
                    (mono, mono)
                } else {
                    (read(0), read(1))
                };
                if channels == 1 {
                    frame[0] += (l * left + r * right) * 0.5;
                } else {
                    frame[0] += l * left;
                    frame[1] += r * right;
                }
                voice.at += step;
            }
            true
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tone(frames: usize) -> Arc<Clip> {
        Arc::new(Clip {
            rate: 100,
            channels: 1,
            samples: vec![1.0; frames],
        })
    }

    #[test]
    fn a_one_shot_ends_and_resamples_to_the_output_rate() {
        let mut mixer = Mixer::default();
        mixer.apply(Command::Play(VoiceId(0), tone(10), Sound::default()));
        // 10 frames at 100 Hz last 20 frames at 200 Hz.
        let mut out = vec![0.0; 2 * 30];
        mixer.render(&mut out, 2, 200);
        assert_eq!(out[2 * 19], 1.0);
        assert_eq!(out[2 * 20], 0.0);
        assert_eq!(mixer.playing(), 0);
    }

    #[test]
    fn looping_sounds_play_until_stopped() {
        let mut mixer = Mixer::default();
        let looping = Sound {
            looping: true,
            ..Sound::default()
        };
        mixer.apply(Command::Play(VoiceId(1), tone(4), looping));
        let mut out = vec![0.0; 2 * 50];
        mixer.render(&mut out, 2, 100);
        assert!(out.iter().all(|&s| s == 1.0));
        mixer.apply(Command::Stop(VoiceId(1)));
        assert_eq!(mixer.playing(), 0);
    }

    #[test]
    fn spatial_sound_pans_and_fades_with_distance() {
        let mut mixer = Mixer::default();
        let falloff = Some(Falloff {
            near: 1.0,
            far: 10.0,
        });
        let at = |x| Sound {
            spatial: falloff,
            position: Vec3::new(x, 0.0, 0.0),
            looping: true,
            ..Sound::default()
        };
        mixer.apply(Command::Play(VoiceId(2), tone(8), at(-2.0)));
        let mut out = vec![0.0; 2];
        mixer.render(&mut out, 2, 100);
        assert!(out[0] > 0.1 && out[1].abs() < 1e-6, "{out:?}");
        mixer.apply(Command::Move(VoiceId(2), Vec3::new(20.0, 0.0, 0.0)));
        mixer.render(&mut out, 2, 100);
        assert_eq!(out, [0.0, 0.0]);
    }
}
