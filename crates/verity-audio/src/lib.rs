//! Sound playback: decoded [`Clip`]s mixed by a [`Mixer`] on the output
//! device's thread, which [`Audio`] opens and sends changes to.

mod clip;
mod mixer;

use std::sync::{
    Arc,
    mpsc::{self, Sender},
};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use glam::Vec3;

pub use clip::Clip;
pub use mixer::{Command, Falloff, Listener, Mixer, Sound, VoiceId};

/// The default output device, playing whatever is sent to it until dropped.
pub struct Audio {
    _stream: cpal::Stream,
    commands: Sender<Command>,
    next: u64,
}

impl Audio {
    pub fn open() -> Result<Self, String> {
        let device = (cpal::default_host().default_output_device())
            .ok_or("no audio output device")?;
        let supported = device.default_output_config().map_err(|e| e.to_string())?;
        let format = supported.sample_format();
        let config: cpal::StreamConfig = supported.into();
        let (channels, rate) = (config.channels as usize, config.sample_rate.0);
        let (commands, received) = mpsc::channel();
        let mut mixer = Mixer::default();
        // Commands are applied on the audio thread, so the game never waits on it.
        let mut mix = move |out: &mut [f32]| {
            while let Ok(command) = received.try_recv() {
                mixer.apply(command);
            }
            mixer.render(out, channels, rate);
        };
        let error = |e| eprintln!("audio output: {e}");
        let stream = match format {
            cpal::SampleFormat::F32 => device.build_output_stream(
                &config,
                move |out: &mut [f32], _: &_| mix(out),
                error,
                None,
            ),
            cpal::SampleFormat::I16 => {
                let mut scratch = Vec::new();
                device.build_output_stream(
                    &config,
                    move |out: &mut [i16], _: &_| {
                        scratch.resize(out.len(), 0.0);
                        mix(&mut scratch);
                        for (out, s) in out.iter_mut().zip(&scratch) {
                            *out = (s.clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
                        }
                    },
                    error,
                    None,
                )
            }
            other => return Err(format!("unsupported output sample format {other}")),
        }
        .map_err(|e| e.to_string())?;
        stream.play().map_err(|e| e.to_string())?;
        Ok(Self {
            _stream: stream,
            commands,
            next: 0,
        })
    }

    pub fn play(&mut self, clip: Arc<Clip>, sound: Sound) -> VoiceId {
        let id = VoiceId(self.next);
        self.next += 1;
        self.send(Command::Play(id, clip, sound));
        id
    }

    pub fn set_position(&self, voice: VoiceId, position: Vec3) {
        self.send(Command::Move(voice, position));
    }

    pub fn stop(&self, voice: VoiceId) {
        self.send(Command::Stop(voice));
    }

    pub fn set_listener(&self, listener: Listener) {
        self.send(Command::Listener(listener));
    }

    fn send(&self, command: Command) {
        // The stream owns the receiver, so this fails only once it is gone.
        let _ = self.commands.send(command);
    }
}
