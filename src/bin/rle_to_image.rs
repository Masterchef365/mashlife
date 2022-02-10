use anyhow::{Context, Result};
use mashlife::io::cells_to_pixels;
use std::path::PathBuf;
use structopt::StructOpt;

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
    mashlife::io::write_png(
        args.out_path
            .unwrap_or(PathBuf::new())
            .join(format!("{rle_name}.png")),
        &rle,
        rle_width as _,
    )
    .context("Writing image")?;

    Ok(())
}
