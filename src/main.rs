use anyhow::{Context as AnyhowContext, Result};
use idek::prelude::*;
use idek::{prelude::*, IndexBuffer, MultiPlatformCamera};
use mashlife::io::cells_to_pixels;
use mashlife::{Coord, Handle, HashLife, Rect};
use std::path::PathBuf;
use structopt::StructOpt;

#[derive(Debug, StructOpt, Default)]
#[structopt(name = "MashLife", about = "Mashes life")]
struct Opt {
    /// Load an RLE file
    rle: PathBuf,

    /// Number of steps to advance (or beginning step in animation)
    #[structopt(short, long, default_value = "8")]
    steps: usize,

    /// Output file path
    #[structopt(short, long)]
    out_path: Option<PathBuf>,

    /// Animation step stride
    #[structopt(long, default_value = "1")]
    stride: usize,

    /// Number of frames to render
    #[structopt(short, long, default_value = "1")]
    frames: usize,

    /// Use the RLE file as the file name prefix
    #[structopt(short, long)]
    use_rle_prefix: bool,

    /// Use step numbers instead of frame numbers
    #[structopt(long)]
    use_step_numbers: bool,

    /// Use the specified output width, as opposed the default view
    #[structopt(short, long)]
    image_width: Option<i64>,

    /// Visualize with VR
    #[structopt(long)]
    vr: bool,
}

fn prepare_data(args: &Opt) -> Result<(HashLife, Handle, Handle)> {
    /*
    let rle_name = args
        .rle
        .file_stem()
        .context("No file step")?
        .to_str()
        .context("RLE file not utf-8")?;
    */

    // Load RLE
    let (rle, rle_width) = mashlife::io::load_rle(&args.rle).context("Failed to load RLE file")?;
    let rle_height = rle.len() / rle_width;

    let max_rle_dim = rle_height.max(rle_width);
    let largest_num_steps = args.steps + args.stride * (args.frames - 1);
    let n = highest_pow_2(max_rle_dim as _).max(highest_pow_2(largest_num_steps as u64) + 2);

    // Create simulation
    let mut life = HashLife::new();

    // Insert RLE
    let half_width = 1 << n - 1;
    let quarter_width = 1 << n - 2;

    let view_width = args.image_width.unwrap_or(half_width);

    let view_tl = (
        (half_width - view_width as i64) / 2,
        (half_width - view_width as i64) / 2,
    );

    //let view_rect = (view_tl, (view_tl.0 + view_width, view_tl.1 + view_width));

    let insert_tl = (
        (half_width - rle_width as i64) / 2 + quarter_width,
        (half_width - rle_height as i64) / 2 + quarter_width,
    );

    let input_cell = life.insert_array(&rle, rle_width, insert_tl, n as _);

    let result_cell = life.result(input_cell, args.steps, Some(view_tl));

    // Calculate result
    /*
    for (frame_idx, steps) in (args.steps..)
        .step_by(args.stride)
        .take(args.frames)
        .enumerate()
    {
        let begin_time = std::time::Instant::now();

        let handle = life.result(handle, steps, Some((0, 0)));

        let elapsed = begin_time.elapsed();
        println!("{}: {}ms", steps, elapsed.as_secs_f32() * 1e3);

        if let Some(out_path) = &args.out_path {
            let raster = life.raster(handle, view_rect);

            // Name image
            let frame_num = if args.use_step_numbers {
                steps
            } else {
                frame_idx
            };

            let image_name = if args.use_rle_prefix {
                format!("{}_{}.png", rle_name, frame_num)
            } else {
                format!("{}.png", frame_num)
            };

            // Write image
            let pixels = cells_to_pixels(&raster);
            mashlife::io::write_png(out_path.join(image_name), &pixels, view_width as _)
                .context("Writing image")?;
        }
    }
    */

    Ok((life, input_cell, result_cell))
}

fn scale_transform(sc: f32) -> [[f32; 4]; 4] {
    [
        [sc, 0., 0., 0.],
        [0., sc, 0., 0.],
        [0., 0., sc, 0.],
        [0., 0., 0., 1.],
    ]
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

/// Push a mesh to the graphicsbuilder containing the true cells in the given array with size 2^n
fn raster_to_mesh(b: &mut GraphicsBuilder, data: &[bool], n: usize, color: [f32; 3], y: f32) {
    // TODO: Run-length compression for faces
    let width = 1 << n;
    assert_eq!(data.len(), width * width);
    for (row_idx, row) in data.chunks_exact(width).enumerate() {
        for (col_idx, &elem) in row.iter().enumerate() {
            if elem {
                let (x, z) = (row_idx as i64, col_idx as i64);

                let mut push =
                    |x, z| b.push_vertex(Vertex::new(world_to_graphics((x, z), y), color));

                let [tl, tr, bl, br] = [
                    push(x, z),
                    push(x + 1, z),
                    push(x, z + 1),
                    push(x + 1, z + 1),
                ];

                b.push_double_sided(&[tl, tr, bl, tr, br, bl]);
            }
        }
    }
}

#[derive(Default, Clone, Debug)]
pub struct GraphicsBuilder {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
}

impl GraphicsBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    /// Push a Vertex and return it's index
    pub fn push_vertex(&mut self, v: Vertex) -> u32 {
        let idx: u32 = self
            .vertices
            .len()
            .try_into()
            .expect("Vertex limit exceeded");
        self.vertices.push(v);
        idx
    }

    /// Push an index
    pub fn push_index(&mut self, idx: u32) {
        self.indices.push(idx);
    }

    /// Erase all content
    pub fn clear(&mut self) {
        self.indices.clear();
        self.vertices.clear();
    }

    /// Push the given vertices, and their opposite face
    pub fn push_double_sided(&mut self, indices: &[u32]) {
        self.indices.extend_from_slice(indices);
        self.indices.extend(
            indices
                .chunks_exact(3)
                .map(|face| [face[2], face[1], face[0]])
                .flatten(),
        );
    }
}

pub fn draw_rect(b: &mut GraphicsBuilder, ((x1, y1), (x2, y2)): Rect, color: [f32; 3], y: f32) {
    let mut push_worldcoord =
        |pos: Coord| b.push_vertex(Vertex::new(world_to_graphics(pos, y), color));

    let tl = push_worldcoord((x1, y1));
    let tr = push_worldcoord((x2 + 1, y1));
    let bl = push_worldcoord((x1, y2 + 1));
    let br = push_worldcoord((x2 + 1, y2 + 1));

    b.indices
        .extend_from_slice(&[tl, tr, tr, br, br, bl, bl, tl]);
}

pub fn world_to_graphics((x, z): Coord, y: f32) -> [f32; 3] {
    [x as f32, y, z as f32]
}

fn main() -> Result<()> {
    let args = Opt::from_args();
    launch::<Opt, HashlifeVisualizer>(Settings::default().vr(args.vr).args(args))
}

struct HashlifeVisualizer {
    line_verts: VertexBuffer,
    line_indices: IndexBuffer,
    line_shader: Shader,

    tri_verts: VertexBuffer,
    tri_indices: IndexBuffer,
    tri_shader: Shader,

    camera: MultiPlatformCamera,
}

impl App<Opt> for HashlifeVisualizer {
    fn init(ctx: &mut Context, platform: &mut Platform, args: Opt) -> Result<Self> {
        let mut line_builder = GraphicsBuilder::new();
        draw_rect(&mut line_builder, ((0, 0), (10, 10)), [1.; 3], 0.);

        let mut tri_builder = GraphicsBuilder::new();
        raster_to_mesh(&mut tri_builder, &[false, true, true, false], 1, [1.; 3], 10.);

        Ok(Self {
            line_verts: ctx.vertices(&line_builder.vertices, false)?,
            line_indices: ctx.indices(&line_builder.indices, false)?,
            line_shader: ctx.shader(
                DEFAULT_VERTEX_SHADER,
                DEFAULT_FRAGMENT_SHADER,
                Primitive::Lines,
            )?,

            tri_verts: ctx.vertices(&tri_builder.vertices, false)?,
            tri_indices: ctx.indices(&tri_builder.indices, false)?,
            tri_shader: ctx.shader(
                DEFAULT_VERTEX_SHADER,
                DEFAULT_FRAGMENT_SHADER,
                Primitive::Triangles,
            )?,
            camera: MultiPlatformCamera::new(platform),
        })
    }

    fn frame(&mut self, _ctx: &mut Context, _: &mut Platform) -> Result<Vec<DrawCmd>> {
        let scale = scale_transform(0.1);
        Ok(vec![
            DrawCmd::new(self.line_verts)
                .indices(self.line_indices)
                .shader(self.line_shader)
                .transform(scale),
            DrawCmd::new(self.tri_verts)
                .indices(self.tri_indices)
                .shader(self.tri_shader)
                .transform(scale),
        ])
    }

    fn event(
        &mut self,
        ctx: &mut Context,
        platform: &mut Platform,
        mut event: Event,
    ) -> Result<()> {
        if self.camera.handle_event(&mut event) {
            ctx.set_camera_prefix(self.camera.get_prefix())
        }
        idek::close_when_asked(platform, &event);
        Ok(())
    }
}
