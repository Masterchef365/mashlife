use anyhow::{Context, Result};
use mashlife::HashLife;
use std::path::PathBuf;
use structopt::StructOpt;

#[derive(Debug, StructOpt)]
#[structopt(name = "MashLife", about = "Mashes life")]
struct Opt {
    /// Load an RLE file
    rle: PathBuf,

    /// Number of steps to advance (or beginning step in animation)
    #[structopt(short, long, default_value = "8")]
    steps: usize,

    /// Output file path
    #[structopt(short, long, default_value = "")]
    out_path: PathBuf,

    /// Animation step stride
    #[structopt(short, long, default_value = "1")]
    stride: usize,

    /// Number of frames to render
    #[structopt(short, long, default_value = "1")]
    frames: usize,

    /// Use the RLE file as the file name prefix
    #[structopt(short, long)]
    use_rle_prefix: bool,

    /// Use step numbers instead of frame numbers
    #[structopt(long)]
    use_step_numbers: bool
}

fn main() -> Result<()> {
    let args = Opt::from_args();

    let rle_name = args
        .rle
        .file_stem()
        .expect("No file step")
        .to_str()
        .expect("RLE file not utf-8");

    // Load RLE
    let (rle, rle_width) = mashlife::io::load_rle(&args.rle).context("Failed to load RLE file")?;
    let rle_height = rle.len() / rle_width;

    let max_rle_dim = rle_height.max(rle_width);
    let largest_num_steps = args.steps + args.stride * (args.frames - 1);
    let n = highest_pow_2(max_rle_dim as _).max(highest_pow_2(largest_num_steps as u64) + 2);
    dbg!(n);

    // Create simulation
    let mut life = HashLife::new();

    // Insert RLE
    let center = 1 << n - 1;
    let handle = life.insert_array(&rle, rle_width, (center, center), n as _);

    // Calculate result
    for (frame_idx, steps) in (args.steps..).step_by(args.stride).take(args.frames).enumerate() {
        dbg!(steps);
        let begin_time = std::time::Instant::now();

        let handle = life.result(handle, steps);

        let elapsed = begin_time.elapsed();
        println!("{}: {}ms", steps, elapsed.as_secs_f32() * 1e3);

        // Rasterize result
        let view_rect = ((0, 0), (1 << n, 1 << n));
        let raster = life.raster(handle, view_rect);

        // Name image
        let frame_num = if args.use_step_numbers {
            steps
        } else {
            frame_idx
        };

        let image_name = if args.use_rle_prefix {
            format!("{}_{}.ppm", rle_name, frame_num)
        } else {
            format!("{}.ppm", frame_num)
        };

        // Write image
        let (view_width, _) = mashlife::rect_dimensions(view_rect);
        let pixels = cells_to_pixels(&raster);
        mashlife::io::write_ppm(args.out_path.join(image_name), &pixels, view_width as _)
            .context("Writing image")?;
    }

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
