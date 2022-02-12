use anyhow::{Context as AnyhowContext, Result};
use idek::prelude::*;
use idek::{prelude::*, IndexBuffer, MultiPlatformCamera};
use mashlife::io::cells_to_pixels;
use mashlife::{Coord, Handle, HashLife, Rect, Rules};
use rand::prelude::*;
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

    /// Animation step stride
    #[structopt(long, default_value = "1")]
    stride: usize,

    /// Use the specified output width, as opposed the default view
    #[structopt(short, long)]
    image_width: Option<i64>,

    /// Visualize with VR
    #[structopt(long)]
    vr: bool,

    /// Show quadtree rectangles
    #[structopt(long)]
    rects: bool,

    /// Rule to execute
    #[structopt(short, long, default_value="B3/S23")]
    rule: Rules,

    /// Maximum number of macrocells to visualize
    #[structopt(short, long, default_value="99999")]
    max_vis_cells: usize
}

fn prepare_data(args: &Opt) -> Result<(HashLife, Handle, Handle, Rect)> {
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
    let n = highest_pow_2(max_rle_dim as _).max(highest_pow_2(args.steps as u64) + 2);

    // Create simulation
    let mut life = HashLife::new(args.rule);

    // Insert RLE
    let half_width = 1 << n - 1;
    let quarter_width = 1 << n - 2;

    let view_width = args.image_width.unwrap_or(half_width);

    let view_tl = (
        (half_width - view_width as i64) / 2,
        (half_width - view_width as i64) / 2,
    );

    let view_rect = (view_tl, (view_tl.0 + view_width, view_tl.1 + view_width));

    let insert_tl = (
        (half_width - rle_width as i64) / 2 + quarter_width,
        (half_width - rle_height as i64) / 2 + quarter_width,
    );

    let input_cell = life.insert_array(&rle, rle_width, insert_tl, n as _);

    let result_cell = life.result(input_cell, args.steps, (-quarter_width, -quarter_width), 0);

    /*
    let insert_rect = (
        insert_tl,
        (
            insert_tl.0 + rle_width as i64,
            insert_tl.1 + rle_height as i64,
        ),
    );
    */

    Ok((life, input_cell, result_cell, view_rect))
}

fn scale_transform(xz_scale: f32, y_scale: f32, x: f32, y: f32, z: f32) -> [[f32; 4]; 4] {
    [
        [xz_scale, 0., 0., 0.],
        [0., y_scale, 0., 0.],
        [0., 0., xz_scale, 0.],
        [x * xz_scale, y * y_scale, z * xz_scale, 1.],
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
fn raster_to_mesh(b: &mut GraphicsBuilder, data: &[bool], rect: Rect, color: [f32; 3], y: f32) {
    let ((x1, y1), (x2, y2)) = rect;
    let width = x2 - x1;
    let height = y2 - y1;

    // TODO: Run-length compression for faces
    //dbg!(data.len(), width, height, width * height);
    assert_eq!(data.len(), (width * height) as usize);
    for (row_idx, row) in data.chunks_exact(width as usize).enumerate() {
        for (col_idx, &elem) in row.iter().enumerate() {
            if elem {
                let (x, z) = (col_idx as i64 + x1, row_idx as i64 + y1);

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
    let tr = push_worldcoord((x2, y1));
    let bl = push_worldcoord((x1, y2));
    let br = push_worldcoord((x2, y2));

    b.indices
        .extend_from_slice(&[tl, tr, tr, br, br, bl, bl, tl]);
}

pub fn world_to_graphics((x, z): Coord, y: f32) -> [f32; 3] {
    [x as f32, y, z as f32]
}

pub fn time_to_graphics(time: usize) -> f32 {
    -(time as f32).max(1e-5).log2()
    //time as f32
}

fn main() -> Result<()> {
    let args = Opt::from_args();
    launch::<Opt, HashlifeVisualizer>(Settings::default().vr(args.vr).args(args))
}

struct ExpandableMesh {
    vertices: VertexBuffer,
    indices: IndexBuffer,
    index_capacity: usize,
    vertex_capacity: usize,
    index_count: u32,
}

const ZERO_VERTEX: Vertex = Vertex {
    pos: [0.; 3],
    color: [0.; 3],
};

impl ExpandableMesh {
    pub fn new(ctx: &mut Context) -> Result<Self> {
        Self::with_capacity(ctx, 1024, 1024)
    }

    pub fn with_capacity(ctx: &mut Context, vertices: usize, indices: usize) -> Result<Self> {
        Ok(Self {
            vertices: ctx.vertices(&vec![ZERO_VERTEX; vertices], true)?,
            indices: ctx.indices(&vec![0; indices], true)?,
            index_capacity: indices,
            vertex_capacity: vertices,
            index_count: 0,
        })
    }

    pub fn update_indices(&mut self, ctx: &mut Context, indices: &[u32]) -> Result<()> {
        self.index_count = indices.len() as u32;
        if indices.len() > self.index_capacity {
            self.index_capacity = 2 * indices.len(); // That oughtta hold em

            // TODO: Delete the old buffer!

            let mut new_indices = vec![0; self.index_capacity];
            new_indices[..indices.len()].copy_from_slice(indices);
            self.indices = ctx.indices(&new_indices, true)?;
            Ok(())
        } else {
            ctx.update_indices(self.indices, indices)
        }
    }

    pub fn update_vertices(&mut self, ctx: &mut Context, vertices: &[Vertex]) -> Result<()> {
        if vertices.len() > self.vertex_capacity {
            self.vertex_capacity = 2 * vertices.len(); // That oughtta hold em

            // TODO: Delete the old buffer!

            let mut new_vertices = vec![ZERO_VERTEX; self.vertex_capacity];
            new_vertices[..vertices.len()].copy_from_slice(vertices);
            self.vertices = ctx.vertices(&new_vertices, true)?;
            Ok(())
        } else {
            ctx.update_vertices(self.vertices, vertices)
        }
    }

    pub fn update_from_graphics_builder(
        &mut self,
        ctx: &mut Context,
        b: &GraphicsBuilder,
    ) -> Result<()> {
        self.update_vertices(ctx, &b.vertices)?;
        self.update_indices(ctx, &b.indices)
    }

    pub fn draw(&self) -> DrawCmd {
        DrawCmd::new(self.vertices)
            .indices(self.indices)
            .limit(self.index_count)
    }
}

struct HashlifeVisualizer {
    lines: ExpandableMesh,
    tris: ExpandableMesh,

    tri_shader: Shader,
    line_shader: Shader,

    camera: MultiPlatformCamera,

    frame: usize,

    args: Opt,

    scale: [[f32; 4]; 4],
}

impl App<Opt> for HashlifeVisualizer {
    fn init(ctx: &mut Context, platform: &mut Platform, args: Opt) -> Result<Self> {
        // Calculate
        let (life, input_cell, result_cell, view_rect) = prepare_data(&args)?;

        let (line_builder, tri_builder, scale) =
            draw_cells(&life, input_cell, result_cell, view_rect, &args);

        let mut lines = ExpandableMesh::new(ctx)?;
        let mut tris = ExpandableMesh::new(ctx)?;

        lines.update_from_graphics_builder(ctx, &line_builder)?;
        tris.update_from_graphics_builder(ctx, &tri_builder)?;

        Ok(Self {
            scale,
            lines,
            tris,
            args,

            line_shader: ctx.shader(
                DEFAULT_VERTEX_SHADER,
                DEFAULT_FRAGMENT_SHADER,
                Primitive::Lines,
            )?,

            frame: 0,

            tri_shader: ctx.shader(
                DEFAULT_VERTEX_SHADER,
                DEFAULT_FRAGMENT_SHADER,
                Primitive::Triangles,
            )?,

            camera: MultiPlatformCamera::new(platform),
        })
    }

    fn frame(&mut self, ctx: &mut Context, _: &mut Platform) -> Result<Vec<DrawCmd>> {
        self.frame += 1;

        if self.frame % 10 == 0 && self.args.stride != 0 {
            // Calculate
            let (life, input_cell, result_cell, view_rect) = prepare_data(&self.args)?;

            let (line_builder, tri_builder, scale) =
                draw_cells(&life, input_cell, result_cell, view_rect, &self.args);

            dbg!(line_builder.vertices.len());
            dbg!(line_builder.indices.len());
            dbg!(tri_builder.vertices.len());
            dbg!(tri_builder.indices.len());

            self.lines
                .update_from_graphics_builder(ctx, &line_builder)?;
            self.tris.update_from_graphics_builder(ctx, &tri_builder)?;
            self.scale = scale;

            self.args.steps += self.args.stride;
        }

        Ok(vec![
            self.lines
                .draw()
                .shader(self.line_shader)
                .transform(self.scale),
            self.tris
                .draw()
                .shader(self.tri_shader)
                .transform(self.scale),
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

        use idek::winit::event::{
            ElementState, Event as WinitEvent, KeyboardInput, VirtualKeyCode, WindowEvent,
        };
        if let Event::Winit(WinitEvent::WindowEvent {
            event:
                WindowEvent::KeyboardInput {
                    input:
                        KeyboardInput {
                            state,
                            virtual_keycode,
                            ..
                        },
                    ..
                },
            ..
        }) = event
        {
            if *state == ElementState::Pressed {
                //if virtual_keycode = Some(VirtualKeyCode::
            }
        }

        Ok(())
    }
}

fn square_rect(corner: i64, width: i64) -> Rect {
    ((corner, corner), (corner + width, corner + width))
}

fn mix(a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
    let mut output = [0.; 3];
    output
        .iter_mut()
        .zip(a)
        .zip(b)
        .for_each(|((o, a), b)| *o = a * (1. - t) + b * t);
    output
}

fn draw_cells(
    life: &HashLife,
    input_cell: Handle,
    _result_cell: Handle,
    _view_rect: Rect,
    args: &Opt,
) -> (GraphicsBuilder, GraphicsBuilder, [[f32; 4]; 4]) {
    let mut line_builder = GraphicsBuilder::new();
    let mut tri_builder = GraphicsBuilder::new();

    draw_rect(
        &mut line_builder,
        square_rect(0, life.macrocell(input_cell).n as _),
        [1.; 3],
        0.,
    );

    let want = args.max_vis_cells;
    let have = life.macrocells().count();
    let p = 1. - (want as f64 / have as f64).clamp(0., 1.);
    dbg!(want, have, p);

    let mut rng = rand::thread_rng();

    // Draw steps
    for (handle, cell) in life.macrocells() {
        let width = 1 << cell.n;
        if let Some((tl, time)) = cell.creation_coord {
            let t = time as f32 / args.steps as f32;
            let w = 0.3;
            let t = t.sqrt() * (1. + w) - w;

            //
            //if (time as f32).log2() as u32 % 5 != 0 {
            if rng.gen_bool(p) {
                continue;
            }

            let mut level_rng = rand::rngs::SmallRng::seed_from_u64(time as u64);
            let t = t + level_rng.gen_range(-1.0..=1.0) * 0.25;

            let color = mix(
                mix([1.; 3], [0.823, 0.162, 1.000], t),
                mix([0.881, 0.190, 0.990], [0.; 3], t),
                t,
            );

            //if level_rng.gen_bool(0.1) {
            /*
            if rand::thread_rng().gen_bool(0.1) {
            color = [0.; 3];
            }
            */

            let rect = (tl, (tl.0 + width, tl.1 + width));

            let y = time_to_graphics(time);

            if args.rects {
                draw_rect(&mut line_builder, rect, color, y);
            }

            let raster = life.raster(handle, square_rect(0, width));
            raster_to_mesh(&mut tri_builder, &raster, rect, color, y);
        }
    }

    // Draw input
    let input_n = life.macrocell(input_cell).n;
    let input_rect = square_rect(0, 1 << input_n);
    let input_raster = life.raster(input_cell, input_rect);

    raster_to_mesh(
        &mut tri_builder,
        &input_raster,
        input_rect,
        [1.; 3],
        time_to_graphics(1),
    );

    let half_width = 1 << input_n - 1;

    let offset = half_width as f32;
    let scale = scale_transform(0.1, 3., -offset, 0., -offset);

    // Draw result
    //let result_raster = life.raster(result_cell, view_rect);
    /*
    let result_raster = life.raster(result_cell, view_rect);

    raster_to_mesh(
    &mut tri_builder,
    &result_raster,
    view_rect,
    [1.; 3],
    time_to_graphics(args.steps),
    );
    */

    (line_builder, tri_builder, scale)
}
