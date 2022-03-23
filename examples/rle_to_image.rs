use anyhow::{Context, Result};
use std::path::{PathBuf, Path};
use structopt::StructOpt;
use std::fs::File;
use std::io::BufWriter;

#[derive(Debug, StructOpt)]
#[structopt(name = "MashLife", about = "Mashes life")]
struct Opt {
    /// Load an RLE file
    rle: PathBuf,

    /// Output file path
    #[structopt(short, long)]
    out_path: Option<PathBuf>,
}

fn main() -> Result<()> {
    let args = Opt::from_args();

    let rle_name = args
        .rle
        .file_stem()
        .context("No file step")?
        .to_str()
        .context("RLE file not utf-8")?;

    // Load RLE
    let (rle, rle_width) = mashlife::io::load_rle(&args.rle).context("Failed to load RLE file")?;

    let rle = cells_to_pixels(&rle);

    // Write image
    write_png(
        args.out_path
            .unwrap_or(PathBuf::new())
            .join(format!("{rle_name}.png")),
        &rle,
        rle_width as _,
    )
    .context("Writing image")?;

    Ok(())
}

pub fn write_png(path: impl AsRef<Path>, data: &[u8], width: usize) -> Result<()> {
    assert!(
        data.len() % 3 == 0,
        "Data length must be a multiple of 3 (RGB)"
    );
    let n_pixels = data.len() / 3;

    assert!(
        n_pixels % width == 0,
        "Pixel count must be a multiple of width"
    );
    let height = n_pixels / width;

    let file = File::create(path)?;
    let file = BufWriter::new(file);

    let mut encoder = png::Encoder::new(file, width as _, height as _); // Width is 2 pixels and height is 1.
    encoder.set_color(png::ColorType::Rgb);
    encoder.set_depth(png::BitDepth::Eight);
    let mut writer = encoder.write_header()?;

    writer.write_image_data(&data)?;

    Ok(())
}

pub fn cells_to_pixels(cells: &[bool]) -> Vec<u8> {
    cells
        .iter()
        .map(|&b| [b as u8 * 255; 3])
        .flatten()
        .collect()
}
