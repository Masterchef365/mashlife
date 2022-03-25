use anyhow::{Context, Result};
use mashlife::{HashLife, Rules};
use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};
use structopt::StructOpt;

#[derive(Debug, StructOpt)]
#[structopt(name = "MashLife", about = "Mashes life")]
struct Opt {
    /// Load an RLE file
    rle: PathBuf,

    /// Output file path, defaults to stdout
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
    let rle_height = rle.len() / rle_width;

    let mut life = HashLife::new(Rules::default());
    let n = 20;

    let half_width = 1 << (n - 1);

    life.insert_array(&rle, rle_width, (half_width, half_width), n);

    if let Some(out_path) = args.out_path {
        write_child_graph(&life, BufWriter::new(File::create(out_path)?))?;
    } else {
        write_child_graph(&life, std::io::stdout())?;
    }

    Ok(())
}

fn write_child_graph<W: Write>(life: &HashLife, mut f: W) -> io::Result<()> {
    writeln!(f, "strict digraph {{")?;
    for (idx, cell) in life.cells().iter().enumerate() {
        for child in cell.children {
            writeln!(f, "    {} -> {}", idx, child.id())?;
        }
    }
    writeln!(f, "}}")
}
