pub mod io;
pub mod geometry;
mod rules;
pub use rules::Rules;
use geometry::*;
use std::collections::HashMap;
type ZwoHasher = std::hash::BuildHasherDefault<zwohash::ZwoHasher>;

// TODO: This assumes you are only using one HashLife instance!!!
/// Handle representing a macrocell
#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct Handle(pub(crate) usize);

/// Dead macrocell. Children are dead cells too
const DEAD: Handle = Handle(0);

/// Living cell
const ALIVE: Handle = Handle(1);

/// Handle chosen to panic in debug mode
const INVALID_HANDLE: Handle = Handle(usize::MAX);

/// A list of four sub-macrocells in row-major order
pub type SubCells = [Handle; 4];

/// A macrocell, which is a list of sub-cells and a size
#[derive(Debug, Copy, Clone)]
pub struct MacroCell {
    /// The width of this cell is 2^n
    pub n: usize,
    /// Indices of child cells (each with width 2^(n-1))
    pub children: SubCells,
}

#[derive(Clone)]
/// An implementation of HashLife;
/// A Cellular-automata acceleration structure
pub struct HashLife {
    /// Mapping from sub-cells to parent cell
    parent_cell: HashMap<SubCells, Handle, ZwoHasher>,
    /// Array of macrocells (mapping cell idx to 
    macrocells: Vec<MacroCell>,
    /// Mapping from time step and handle to result handle
    result: HashMap<(usize, Handle), Handle, ZwoHasher>,
    /// Ruleset
    rules: Rules,
}

impl HashLife {
    /// Create a new HashLife instance with the given rules
    pub fn new(rules: Rules) -> Self {
        Self {
            rules,
            parent_cell: [([DEAD; 4], DEAD)].into_iter().collect(),
            macrocells: vec![
                // Handle(0) is infinite dead cells.
                // Zero is special; it has children which are all zeros. This makes construction of
                // dead cells very easy, but admits some special cases.
                MacroCell {
                    n: 0,
                    children: [DEAD; 4],
                },
                // Handle(1) is always a single, live cell.
                MacroCell {
                    n: 0,
                    children: [INVALID_HANDLE; 4],
                },
            ],
            result: Default::default(),
        }
    }

    /// Insert the given data with the given width and top-left
    /// corner and return a macrocell of width 2^n, padded with
    /// zeroes everywhere else. Top-left is relative to the top-left of the cell.
    pub fn insert_array(
        &mut self,
        input: &[bool],
        width: usize,
        tl_corner: Coord,
        n: usize,
    ) -> Handle {
        // Calculate height
        assert_eq!(
            input.len() % width,
            0,
            "Data length ({}) must be a multiple of width ({})",
            input.len(),
            width
        );
        let height = input.len() / width;

         // Calculate input rect
        let (left, top) = tl_corner;
        let br_corner = (left + width as i64, top + height as i64);
        let rect = (tl_corner, br_corner);

        self.insert_rect(input, (0, 0), rect, n)
    }

    /// Insert the given array with dimensions described by input_rect
    /// Positions within the input rect are calculated relative to tl_corner, representing the
    /// top-left corner of the handle in the space of the rect
    /// n is the 
    fn insert_rect(
        &mut self,
        input: &[bool],
        tl_corner: Coord,
        input_rect: Rect,
        n: usize,
    ) -> Handle {
        // Return the input pixel at the given coordinates
        if n == 0 {
            return Handle(
                sample_rect(tl_corner, input_rect)
                    .map(|idx| input[idx])
                    .unwrap_or(false) as usize,
            );
        }

        // Short circuit for zeroes
        if !rect_intersect(rect_pow_2(tl_corner, n), input_rect) {
            return DEAD;
        }

        // Calculate which macrocell we are in
        let children = subcoords(tl_corner, n - 1)
            .map(|sub_corner| self.insert_rect(input, sub_corner, input_rect, n - 1));

        //self.insert_cell(children, n, Some((tl_corner, 0)))
        self.parent(children, n)
    }

    /// Return the parent handle of the given cells, or create a new parent with the given size n
    fn parent(
        &mut self,
        children: SubCells,
        n: usize,
    ) -> Handle {
        match self.parent_cell.get(&children) {
            None => {
                let idx = self.macrocells.len();
                self.macrocells.push(MacroCell {
                    n,
                    children,
                });
                let handle = Handle(idx);
                self.parent_cell.insert(children, handle);
                handle
            }
            Some(idx) => *idx,
        }
    }

    /// Returns the given MacroCell advanced by the given number of steps `dt` (up to and including
    /// 2^(n-2) where 2^n is the width of the cell)
    pub fn result(&mut self, handle: Handle, dt: usize, corner: Coord) -> Handle {
        // Fast-path for all-dead cells
        if handle == DEAD {
            return handle;
        }

        let cell = self.macrocell(handle);
        let cell_n = cell.n;

        assert!(
            cell.n >= 2,
            "Results can only be computed for 4x4 cells and larger"
        );

        assert!(dt <= 1 << cell.n - 2, "dt ({}) must be <= 2^(n - 2), n={}", dt, cell.n);

        // Check if we already know the result
        if let Some(&result) = self.result.get(&(dt, handle)) {
            return result;
        }

        let (cx, cy) = corner;

        // Solve 4x4 if we're at n = 2
        let result = if cell_n == 2 {
            let result = match dt {
                0 => self.center_passthrough(handle),
                1 => solve_4x4(self.grandchildren(handle), &self.rules),
                _ => panic!("Invalid dt for n = 2"),
            };
            self.parent(result, cell_n - 1)
        } else {
            let sub_step_dt = 1 << cell_n - 3;

            let dt_1 = dt.min(sub_step_dt);
            let dt_2 = dt.checked_sub(sub_step_dt).unwrap_or(0);

            /*
            Deconstruct the quadrants of the macrocell, like so:
            | _ B | E _ |
            | C D | G H |
            +-----+-----+
            | I J | M N |
            | _ L | O _ |
            */
            let [
                tl @ [_, b, c, d], // Top left
                tr @ [e, _, g, h], // Top right
                bl @ [i, j, _, l], // Bottom left
                br @ [m, n, o, _] // Bottom right
            ] = self.grandchildren(handle);

            let grandchild_width = 1i64 << cell_n - 2; // Width of a grandchild
            let great_grandchild_width = 1i64 << cell_n - 3; // Half the width of a grandchild

            let corner_3x3 = |u, v| (great_grandchild_width + u * grandchild_width + cx, great_grandchild_width + v * grandchild_width + cy);

            let middle_3x3 = [
                // Top inner row
                (corner_3x3(0, 0), tl),
                (corner_3x3(1, 0), [b, e, d, g]),
                (corner_3x3(2, 0), tr),
                // Middle inner row
                (corner_3x3(0, 1), [c, d, i, j]),
                (corner_3x3(1, 1), [d, g, j, m]),
                (corner_3x3(2, 1), [g, h, m, n]),
                // Bottom inner row
                (corner_3x3(0, 2), bl),
                (corner_3x3(1, 2), [j, m, l, o]),
                (corner_3x3(2, 2), br),
            ]
            .map(|(coord, subcells)| (coord, self.parent(subcells, cell_n - 1)));

            /*
            Compute results or passthroughs for grandchild nodes
            | Q R S |
            | T U V |
            | W X Y |
            */

            let [q, r, s, t, u, v, w, x, y] =
                middle_3x3.map(|(coord, handle)| self.result(handle, dt_1, coord));

            let corner_2x2 = |u, v| (grandchild_width + u * grandchild_width + cx, grandchild_width + v * grandchild_width + cy);

            // Get the middle four quadrants of the 3x3 above
            let middle_2x2 = [
                (corner_2x2(0, 0), [q, r, t, u]),
                (corner_2x2(1, 0), [r, s, u, v]),
                (corner_2x2(0, 1), [t, u, w, x]),
                (corner_2x2(1, 1), [u, v, x, y]),
            ]
            .map(|(coord, subcells)| (coord, self.parent(subcells, cell_n - 1)));

            // Compute results or passthroughs for child nodes
            let result =
                middle_2x2.map(|(coord, handle)| self.result(handle, dt_2, coord));

            // Save the result
            self.parent(result, cell_n - 1)
        };

        self.result.insert((dt, handle), result);
        result
    }

    /// Return the macrocell behind the given handle
    fn macrocell(&self, Handle(idx): Handle) -> MacroCell {
        self.macrocells[idx]
    }

    pub fn cells(&self) -> &[MacroCell] {
        &self.macrocells
    }

    /// Get the center 4 cells of the given cell
    fn center_passthrough(&mut self, handle: Handle) -> SubCells {
        let [
            [_, _, _, d], // Top left
            [_, _, g, _], // Top right
            [_, j, _, _], // Bottom left
            [m, _, _, _] // Bottom right
        ] = self.grandchildren(handle);
        [d, g, j, m]
    }

    /// Grandchildren of the given handle
    fn grandchildren(&self, Handle(parent): Handle) -> [SubCells; 4] {
        self.macrocells[parent]
            .children
            .map(|Handle(child)| self.macrocells[child].children)
    }

    /// Create a raster image from the given node
    pub fn raster(&self, root: Handle, rect: Rect, min_n: usize) -> Vec<bool> {
        let (width, height) = rect_dimensions(rect);
        let mut image = vec![false; (width * height) as usize];

        let mut set_pixel = |pos: Coord| {
            if let Some(idx) = sample_rect(pos, rect) {
                image[idx] = true;
            }
        };

        self.resolve((0, 0), &mut set_pixel, min_n, rect, root);

        image
    }

    /// Resolve the given handle into an image by calling the given function with pixel
    /// coordinates. min_n specifies the minimum macrocell size/depth the function will traverse
    pub fn resolve(
        &self,
        corner: Coord,
        image: &mut impl FnMut(Coord),
        min_n: usize,
        rect: Rect,
        handle: Handle,
    ) {
        // Check if the output is gauranteed to be zero (assumes input is zeroed!)
        if handle == DEAD {
            return;
        }

        let cell = self.macrocell(handle);
        debug_assert_ne!(cell.n, 0);

        // Check if this macrocell is in view
        let width = 1 << cell.n;
        let cell_rect = (corner, (corner.0 + width, corner.1 + width));
        if !rect_intersect(cell_rect, rect) {
            return;
        }

        // If at the base layer, output to the image. Otherwise compute sub-cells
        if cell.n == min_n {
            image(corner);
        } else {
            for (sub_corner, node) in subcoords(corner, cell.n - 1).into_iter().zip(cell.children) {
                self.resolve(sub_corner, image, min_n, rect, node);
            }
        }
    }

    /// Returns a new handle with the cell at `coord` set to `value` in `handle`.
    /// If the given cell is empty, return a new cell with the given size `n` at the
    /// corresponding `value` set at `coord`
    pub fn modify(
        &mut self,
        handle: Handle,
        coord: Coord,
        value: bool,
        n: usize,
    ) -> Handle {
        let cell = self.macrocell(handle);
        let mut children = cell.children;

        // The dead cell is a special case, as it has no n and we can only insert. Otherwise sanity check.
        if handle != DEAD {
            assert_eq!(n, cell.n);
        }
        
        if n == 0 {
            return match value {
                false => DEAD,
                true => ALIVE,
            };
        }

        let (idx, subcoord) = Self::child_subcoord_idx(coord, n);

        children[idx] = self.modify(children[idx], subcoord, value, n - 1);

        self.parent(children, n)
    }

    /// Return the child index and sub-coordinate at the given coordinates
    fn child_subcoord_idx((x, y): Coord, n: usize) -> (usize, Coord) {
        let half_width = 1 << n - 1;
        match (x >= half_width, y >= half_width) {
            (false, false) => (0, (x, y)),
            (true, false) => (1, (x - half_width, y)),
            (false, true) => (2, (x, y - half_width)),
            (true, true) => (3, (x - half_width, y - half_width)),
        }
    }

    /// Read a single cell at the given coordinates
    pub fn read(&self, handle: Handle, coord: Coord) -> bool {
        match handle {
            DEAD => return false,
            ALIVE => return true,
            other => {
                let cell = self.macrocell(other);
                let (idx, subcoord) = Self::child_subcoord_idx(coord, cell.n);
                self.read(cell.children[idx], subcoord)
            }
        }
    }

    /// Return a new handle padded by zeroes on all sides. Useful for padding result calculations
    /// so that the parent cell size does not shrink with each step (for example, if modified)
    pub fn expand(&mut self, handle: Handle) -> Handle {
        let cell = self.macrocell(handle);
        let [tl, tr, bl, br] = cell.children;

        let z = DEAD;

        let tl = self.parent([
            z, z, // 
            z, tl, // 
        ], cell.n);

        let tr = self.parent([
            z, z, // 
            tr, z, // 
        ], cell.n);

        let bl = self.parent([
            z, bl, // 
            z, z, // 
        ], cell.n);

        let br = self.parent([
            br, z, // 
            z, z, // 
        ], cell.n);

        self.parent([tl, tr, bl, br], cell.n + 1)
    }
}

/// Solve a 4x4 grid, represented as four corners of row-major 2x2 grids
fn solve_4x4(
    [[a, b, c, d], [e, f, g, h], [i, j, k, l], [m, n, o, p]]: [SubCells; 4],
    rules: &Rules,
) -> SubCells {
    [
        solve_3x3([a, b, e, c, d, g, i, j, m], rules),
        solve_3x3([b, e, f, d, g, h, j, m, n], rules),
        solve_3x3([c, d, g, i, j, m, k, l, o], rules),
        solve_3x3([d, g, h, j, m, n, l, o, p], rules),
    ]
}

/// Solve a row-major 3x3 grid
fn solve_3x3([a, b, c, d, e, f, g, h, i]: [Handle; 9], rules: &Rules) -> Handle {
    let count = [a, b, c, d, f, g, h, i]
        .into_iter()
        .filter(|&h| h == ALIVE)
        .count();
    Handle(rules.execute(e == ALIVE, count) as usize)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::str::FromStr;

    #[test]
    fn test_rect_intersect() {
        #[track_caller]
        fn reflexive(a: Rect, b: Rect, expect: bool) {
            assert_eq!(rect_intersect(a, b), expect);
            assert_eq!(rect_intersect(b, a), expect);
        }
        reflexive(((-10, -10), (10, 10)), ((-5, -5), (5, 5)), true);
        reflexive(((-10, -10), (10, 10)), ((0, 0), (5, 5)), true);
        reflexive(((3, 3), (10, 10)), ((0, 0), (5, 5)), true);
        reflexive(((7, 7), (10, 10)), ((0, 0), (5, 5)), false);
        reflexive(((7, 7), (10, 10)), ((0, 0), (5, 5)), false);
        reflexive(((7, 7), (10, 10)), ((-5, -5), (0, 0)), false);
        reflexive(((7, 7), (10, 10)), ((-5, -5), (5, 50)), false);
        reflexive(((0, 0), (10, 10)), ((-5, -5), (5, 50)), true);
        reflexive(((0, 0), (10, 10)), ((-5, -5), (50, 5)), true);

        reflexive(((0, 0), (10, 10)), ((-5, -5), (-2, 50)), false);
        reflexive(((0, 0), (10, 10)), ((-5, -5), (50, -2)), false);
    }

    #[test]
    fn test_solve_4x4() {
        let rules = Rules::from_str("B3/S23").unwrap();
        assert_eq!(
            solve_4x4(
                [
                    [
                        0, 0, //.
                        1, 0 //.
                    ]
                    .map(Handle),
                    [
                        1, 0, //.
                        1, 0, //.
                    ]
                    .map(Handle),
                    [
                        0, 1, //.
                        0, 0, //.
                    ]
                    .map(Handle),
                    [
                        1, 0, //.
                        0, 0, //.
                    ]
                    .map(Handle)
                ],
                &rules
            ),
            [
                0, 1, //.
                1, 1, //.
            ]
            .map(Handle)
        );
    }
}

impl Handle {
    pub fn id(&self) -> usize {
        self.0
    }
}
