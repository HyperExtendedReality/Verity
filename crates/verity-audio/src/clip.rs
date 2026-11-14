//! Decoded sounds.

/// A decoded sound: interleaved samples in -1..=1.
#[derive(Clone, Debug, PartialEq)]
pub struct Clip {
    pub rate: u32,
    pub channels: u16,
    pub samples: Vec<f32>,
}

impl Clip {
    pub fn frames(&self) -> usize {
        self.samples.len() / self.channels as usize
    }

    /// Reads a mono or stereo WAV file of 8, 16, 24 or 32-bit integer or
    /// 32-bit float samples.
    pub fn from_wav(bytes: &[u8]) -> Result<Self, String> {
        if bytes.len() < 12 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
            return Err("not a WAV file".into());
        }
        let u16_at = |b: &[u8], i: usize| u16::from_le_bytes([b[i], b[i + 1]]);
        let u32_at = |b: &[u8], i: usize| u32::from_le_bytes([b[i], b[i + 1], b[i + 2], b[i + 3]]);
        let (mut format, mut data) = (None, None);
        let mut at = 12;
        while at + 8 <= bytes.len() {
            let size = u32_at(bytes, at + 4) as usize;
            let body = &bytes[at + 8..(at + 8).saturating_add(size).min(bytes.len())];
            match &bytes[at..at + 4] {
                b"fmt " if body.len() >= 16 => {
                    let mut tag = u16_at(body, 0);
                    // WAVE_FORMAT_EXTENSIBLE keeps the real tag at the start of its subformat.
                    if tag == 0xFFFE && body.len() >= 26 {
                        tag = u16_at(body, 24);
                    }
                    format = Some((tag, u16_at(body, 2), u32_at(body, 4), u16_at(body, 14)));
                }
                b"data" => data = Some(body),
                _ => {}
            }
            // Chunks are padded to an even size.
            at = at.saturating_add(8 + size + size % 2);
        }
        let (tag, channels, rate, bits) = format.ok_or("WAV file has no format chunk")?;
        let data = data.ok_or("WAV file has no data chunk")?;
        if !(1..=2).contains(&channels) {
            return Err(format!("{channels} channels; only mono and stereo are supported"));
        }
        if rate == 0 {
            return Err("WAV sample rate is zero".into());
        }
        let samples = match (tag, bits) {
            (1, 8) => data.iter().map(|&s| (s as f32 - 128.0) / 128.0).collect(),
            (1, 16) => (data.as_chunks().0.iter())
                .map(|&s| i16::from_le_bytes(s) as f32 / 32768.0)
                .collect(),
            (1, 24) => (data.as_chunks::<3>().0.iter())
                .map(|&[a, b, c]| i32::from_le_bytes([0, a, b, c]) as f32 / 2_147_483_648.0)
                .collect(),
            (1, 32) => (data.as_chunks().0.iter())
                .map(|&s| i32::from_le_bytes(s) as f32 / 2_147_483_648.0)
                .collect(),
            (3, 32) => (data.as_chunks().0.iter())
                .map(|&s| f32::from_le_bytes(s))
                .collect(),
            _ => return Err(format!("unsupported WAV encoding (format {tag}, {bits} bits)")),
        };
        let mut clip = Self {
            rate,
            channels,
            samples,
        };
        // A truncated file may end mid-frame.
        clip.samples.truncate(clip.frames() * channels as usize);
        Ok(clip)
    }
}

#[cfg(test)]
pub(crate) fn wav16(rate: u32, channels: u16, samples: &[i16]) -> Vec<u8> {
    let data: Vec<u8> = samples.iter().flat_map(|s| s.to_le_bytes()).collect();
    let mut out = Vec::new();
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&(36 + data.len() as u32).to_le_bytes());
    out.extend_from_slice(b"WAVEfmt ");
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&channels.to_le_bytes());
    out.extend_from_slice(&rate.to_le_bytes());
    out.extend_from_slice(&(rate * channels as u32 * 2).to_le_bytes());
    out.extend_from_slice(&(channels * 2).to_le_bytes());
    out.extend_from_slice(&16u16.to_le_bytes());
    out.extend_from_slice(b"data");
    out.extend_from_slice(&(data.len() as u32).to_le_bytes());
    out.extend_from_slice(&data);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_16_bit_stereo() {
        let clip = Clip::from_wav(&wav16(22050, 2, &[0, 16384, -32768, 32767])).unwrap();
        assert_eq!((clip.rate, clip.channels, clip.frames()), (22050, 2, 2));
        assert_eq!(clip.samples[..3], [0.0, 0.5, -1.0]);
    }

    #[test]
    fn rejects_other_files() {
        assert!(Clip::from_wav(b"OggS not a wav").is_err());
        let mut surround = wav16(48000, 2, &[0; 6]);
        surround[22] = 6;
        assert!(Clip::from_wav(&surround).is_err());
    }
}
