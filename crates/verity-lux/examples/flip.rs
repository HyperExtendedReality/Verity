//! HDR-FLIP error maps between saved frames: where and how much a method's
//! image differs from a reference, as a viewer would see it.
//!
//! `cargo run --release -p verity-lux --example flip -- REFERENCE.pfm TEST.pfm...`
//!
//! Reads the PFMs `sponza_gi --bench` saves, such as a `--reference` run's
//! `still15.pfm` against another run's `still0.pfm`..`still15.pfm`. Prints each
//! test's mean, median and 99th-percentile FLIP (0 identical, 1 the most a
//! viewer could notice), writes its Magma heatmap beside it as `.flip.png`, and
//! ends with the mean over the tests. `--ppd=N` sets pixels per degree (default
//! 67, a 0.7 m wide 4K monitor 0.7 m away).
use flip_rs::{DEFAULT_PPD, HdrOptions, RgbImage, Weighting, hdr_flip};
use std::{error::Error, fs, path::Path};

/// A colour PFM (`PF`), rows flipped to top first.
fn read_pfm(path: &Path) -> Result<RgbImage<f32>, Box<dyn Error>> {
    let bytes = fs::read(path)?;
    let mut fields = Vec::new();
    let mut start = 0;
    // Three whitespace-separated header lines: kind, size, scale.
    for (i, &b) in bytes.iter().enumerate() {
        if b == b'\n' {
            fields.push(std::str::from_utf8(&bytes[start..i])?.trim().to_owned());
            start = i + 1;
            if fields.len() == 3 {
                break;
            }
        }
    }
    let [kind, size, scale] = fields.as_slice() else { return Err("short PFM header".into()) };
    if kind != "PF" {
        return Err(format!("{}: not a colour PFM", path.display()).into());
    }
    let mut size = size.split_whitespace().map(str::parse::<usize>);
    let (Some(Ok(width)), Some(Ok(height))) = (size.next(), size.next()) else {
        return Err("bad PFM size".into());
    };
    let little = scale.parse::<f32>()? < 0.0;
    let data = &bytes[start..];
    if data.len() < width * height * 12 {
        return Err(format!("{}: truncated", path.display()).into());
    }
    let mut pixels = vec![0.0; width * height * 3];
    for (row, line) in data.chunks_exact(width * 12).take(height).enumerate() {
        let top = (height - 1 - row) * width * 3;
        for (i, &word) in line.as_chunks::<4>().0.iter().enumerate() {
            let value = if little { f32::from_le_bytes(word) } else { f32::from_be_bytes(word) };
            // FLIP rejects nonfinite input; a NaN pixel is as wrong as black.
            pixels[top + i] = if value.is_finite() { value.max(0.0) } else { 0.0 };
        }
    }
    Ok(RgbImage::new(width, height, pixels)?)
}

fn write_png(path: &Path, image: &RgbImage<f32>) -> Result<(), Box<dyn Error>> {
    let file = fs::File::create(path)?;
    let mut encoder = png::Encoder::new(file, image.width() as u32, image.height() as u32);
    encoder.set_color(png::ColorType::Rgb);
    let bytes: Vec<u8> = image.pixels().iter().map(|v| (v.clamp(0.0, 1.0) * 255.0).round() as u8).collect();
    encoder.write_header()?.write_image_data(&bytes)?;
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let ppd = match args.iter().find_map(|a| a.strip_prefix("--ppd=")) {
        Some(text) => text.parse()?,
        None => DEFAULT_PPD,
    };
    let paths: Vec<&String> = args.iter().filter(|a| !a.starts_with("--")).collect();
    let [reference, tests @ ..] = paths.as_slice() else {
        return Err("usage: flip REFERENCE.pfm TEST.pfm...".into());
    };
    if tests.is_empty() {
        return Err("usage: flip REFERENCE.pfm TEST.pfm...".into());
    }
    let reference = read_pfm(Path::new(reference))?;
    let mut options = HdrOptions::default();
    options.ppd = ppd;
    options.return_exposure_map = false;
    let mut total = 0.0;
    println!("{:<40} {:>8} {:>8} {:>8}", "test", "mean", "median", "p99");
    for test in tests {
        let path = Path::new(test);
        let result = hdr_flip(&reference, &read_pfm(path)?, options)?;
        let map = result.error_map;
        let mean = map.statistics().mean;
        let sorted = map.percentiles()?;
        let median = sorted.percentile(0.5, Weighting::Unweighted)?;
        let p99 = sorted.percentile(0.99, Weighting::Unweighted)?;
        println!("{:<40} {mean:8.4} {median:8.4} {p99:8.4}", path.display());
        write_png(&path.with_extension("flip.png"), &map.colorize()?)?;
        total += mean;
    }
    println!("{:<40} {:8.4}", "MEAN", total / tests.len() as f32);
    Ok(())
}
