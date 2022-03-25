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

    /// Omit zeroes
    #[structopt(short, long)]
    no_zeros: bool,

    /// Output file path, defaults to stdout
    #[structopt(short, long)]
    out_path: Option<PathBuf>,
}

fn main() -> Result<()> {
    let args = Opt::from_args();

    // Load RLE
    let (rle, rle_width) = mashlife::io::load_rle(&args.rle).context("Failed to load RLE file")?;

    let mut life = HashLife::new(Rules::default());
    let n = 20;

    let half_width = 1 << (n - 1);

    life.insert_array(&rle, rle_width, (half_width, half_width), n);

    if let Some(out_path) = args.out_path {
        write_child_graph(&life, BufWriter::new(File::create(out_path)?), args.no_zeros)?;
    } else {
        write_child_graph(&life, std::io::stdout(), args.no_zeros)?;
    }

    Ok(())
}

fn write_child_graph<W: Write>(life: &HashLife, mut f: W, omit_zeros: bool) -> io::Result<()> {
    writeln!(f, "strict digraph {{")?;
    for (idx, cell) in life.cells().iter().enumerate() {
        let names = ["TL", "TR", "BL", "BR"];
        for (child, name) in cell.children.into_iter().zip(names) {
            if !((child.id() == 0) && omit_zeros) {
                writeln!(f, "    {} -> {} [label=\"{}\"]", idx, child.id(), name)?;
            }
        }
    }
    writeln!(f, "}}")
}
