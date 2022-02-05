use std::path::PathBuf;
use structopt::StructOpt;
use anyhow::{Result, Context};

#[derive(Debug, StructOpt)]
#[structopt(name = "MashLife", about = "Mashes life")]
struct Opt {
    /// Load an RLE file
    rle: PathBuf,
    // #[structopt(short = "r", long = "rle")]

    /// Number of steps to advance
    #[structopt(short, long, default_value = "8")]
    steps: usize,

    /// Output file path
    #[structopt(short, long, default_value = "out.ppm")]
    outfile: PathBuf,
}

fn main() -> Result<()> {
    let args = Opt::from_args();

    let (rle, rle_width) = mashlife::io::load_rle(args.rle).context("Failed to load RLE file")?;
    
    let pixels = cells_to_pixels(&rle);
    mashlife::io::write_ppm(args.outfile, &pixels, rle_width).context("Writing image")?;

    Ok(())
}

fn cells_to_pixels(cells: &[bool]) -> Vec<u8> {
    cells.iter().map(|&b| [b as u8 * 255; 3]).flatten().collect()
}
