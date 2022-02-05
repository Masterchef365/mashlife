use anyhow::{Context, Result};
use mashlife::HashLife;
use std::path::PathBuf;
use structopt::StructOpt;

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
    let rle_height = rle.len() / rle_width;

    let max_rle_dim = rle_height.max(rle_width);
    let n = args.steps;//highest_pow_2(max_rle_dim as _);
    dbg!(n);

    let mut life = HashLife::new();
    let side_len_half = 1 << n - 1;

    let handle = life.insert_array(&rle, rle_width, (0, 0), n as _);

    let raster = life.raster(handle, ((0, 0), (1<<n, 1<<n)));

    let pixels = cells_to_pixels(&raster);
    mashlife::io::write_ppm(args.outfile, &pixels, 1<<n).context("Writing image")?;

    Ok(())
}

fn cells_to_pixels(cells: &[bool]) -> Vec<u8> {
    cells
        .iter()
        .map(|&b| [b as u8 * 255; 3])
        .flatten()
        .collect()
}

/// Returns the ceiling of logbase 2 of the given integer
fn highest_pow_2(mut v: u64) -> u32 {
    let mut i = 0;
    for bit in 0..u64::BITS - 1 {
        if v & 1 != 0 {
            i = bit + 1;
        }
        v >>= 1;
    }
    i
}
