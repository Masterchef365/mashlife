pub mod io;

use std::collections::HashMap;

// TODO: This assumes you are only using one HashLife instance!!!
#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct Handle(pub(crate) usize);

pub type SubCells = [Handle; 4];

pub type Coord = (i64, i64);
pub type Rect = (Coord, Coord);

#[derive(Clone, Debug)]
pub struct MacroCell {
    /// The width of this cell is 2^n
    pub n: usize,
    /// Indices of child cells (each with width 2^(n-1))
    pub children: SubCells,
    /// This macrocell was created from the given coordinate
    pub creation_coord: Option<Coord>,
    /// Mapping from time step to result (if any)
    pub result: HashMap<usize, Handle>,
}

pub struct HashLife {
    /// Mapping from (sub-cells and time step) to parent cell
    parent_cell: HashMap<SubCells, Handle>,
    /// Array of macrocells
    macrocells: Vec<MacroCell>,
}

impl HashLife {
    pub fn new() -> Self {
        Self {
            // Zero always yields zero
            parent_cell: [([Handle(0); 4], Handle(0))].into_iter().collect(),
            macrocells: vec![
                MacroCell {
                    creation_coord: None,
                    n: 0,
                    children: [Handle(0); 4],
                    result: Default::default(),
                },
                MacroCell {
                    creation_coord: None,
                    n: 0,
                    children: [Handle(usize::MAX); 4],
                    result: Default::default(),
                },
            ],
        }
    }

    pub fn macrocells(&self) -> &[MacroCell] {
        &self.macrocells
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

    pub fn insert_rect(
        &mut self,
        input: &[bool],
        tl_corner: Coord,
        input_rect: Rect,
        n: usize,
    ) -> Handle {
        // Short circuit for zeroes
        if zero_input(tl_corner, n, input_rect) {
            return Handle(0);
        }

        // Return the input pixel at the given coordinates
        if n == 0 {
                let idx = sample_rect(tl_corner, input_rect);
                if idx > Some(input.len()) {
                    let (w, h) = crate::rect_dimensions(input_rect);
                    dbg!(tl_corner, input_rect, idx, w, h);
                }
            return Handle(

                sample_rect(tl_corner, input_rect)
                    .map(|idx| input[idx])
                    .unwrap_or(false) as usize,
            );
        }

        // Calculate which macrocell we are in
        let children = subcoords(tl_corner, n - 1)
            .map(|sub_corner| self.insert_rect(input, sub_corner, input_rect, n - 1));

        self.insert_cell(children, n, Some(tl_corner))
    }

    fn insert_cell(&mut self, children: SubCells, n: usize, coord: Option<Coord>) -> Handle {
        match self.parent_cell.get(&children) {
            None => {
                let idx = self.macrocells.len();
                self.macrocells.push(MacroCell {
                    creation_coord: coord,
                    n,
                    children,
                    result: Default::default(),
                });
                let handle = Handle(idx);
                self.parent_cell.insert(children, handle);
                handle
            }
            Some(idx) => *idx,
        }
    }

    /// Returns the given macro`cell` advanced by the given number of `steps` (up to and including
    /// 2^(n-2) where 2^n is the width of the cell
    pub fn result(&mut self, handle: Handle, dt: usize, coord: Option<Coord>) -> Handle {
        // Fast-path for zeroes
        if handle.0 == 0 {
            return handle;
        }

        let cell = self.macrocell(handle);
        let cell_n = cell.n;

        assert!(
            cell.n >= 2,
            "Results can only be computed for 4x4 cells and larger"
        );

        assert!(dt <= 1 << cell.n - 2, "dt must be <= 2^(n - 2)");

        // Check if we already know the result
        if let Some(&result) = cell.result.get(&dt) {
            return result;
        }

        let quarter_width = 1 << cell.n;
        let result_coord = coord.map(|(x, y)| (x + quarter_width, y + quarter_width));

        // Solve 4x4 if we're at n = 2
        let result = if cell_n == 2 {
            let result = match dt {
                0 => self.center_passthrough(handle),
                1 => solve_4x4(self.grandchildren(handle)),
                _ => panic!("Invalid dt for n = 2"),
            };
            self.insert_cell(result, cell_n - 1, result_coord)
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

            let middle_3x3 = [
                // Top inner row
                tl,
                [b, e, d, g],
                tr,
                // Middle inner row
                [c, d, i, j],
                [d, g, j, m],
                [g, h, m, n],
                // Bottom inner row
                bl,
                [j, m, l, o],
                br,
            ]
            .map(|subcells| self.insert_cell(subcells, cell_n - 1, None));

            /*
            Compute results or passthroughs for grandchild nodes
            | Q R S |
            | T U V |
            | W X Y |
            */

            let [q, r, s, t, u, v, w, x, y] = middle_3x3.map(|handle| self.result(handle, dt_1, None));

            // Get the middle four quadrants of the 3x3 above
            let middle_2x2 = [[q, r, t, u], [r, s, u, v], [t, u, w, x], [u, v, x, y]]
                .map(|subcells| self.insert_cell(subcells, cell_n - 1, None));

            // Compute results or passthroughs for child nodes
            let result = middle_2x2.map(|handle| self.result(handle, dt_2, None));

            // Save the result
            self.insert_cell(result, cell_n - 1, result_coord)
        };

        self.macrocell_mut(handle).result.insert(dt, result);
        result
    }

    /// Return the macrocell behind the given handle
    pub fn macrocell(&self, Handle(idx): Handle) -> &MacroCell {
        &self.macrocells[idx]
    }

    /// Return the mutable macrocell behind the given handle
    fn macrocell_mut(&mut self, Handle(idx): Handle) -> &mut MacroCell {
        &mut self.macrocells[idx]
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
    pub fn raster(&self, root: Handle, rect: Rect) -> Vec<bool> {
        let (width, height) = rect_dimensions(rect);
        let mut data = vec![false; (width * height) as usize];
        self.raster_rec((0, 0), &mut data, rect, root);

        data
    }

    /// Recursively create a raster image
    fn raster_rec(&self, corner: Coord, image: &mut [bool], rect: Rect, handle: Handle) {
        // Check if the output is gauranteed to be zero (assumes input is zeroed!)
        if handle == Handle(0) {
            return;
        }

        let cell = self.macrocell(handle);
        debug_assert_ne!(cell.n, 0);

        if cell.n == 1 {
            for (pos, Handle(val)) in subcoords(corner, 0).into_iter().zip(cell.children) {
                if let Some(idx) = sample_rect(pos, rect) {
                    image[idx] = match val {
                        0 => false,
                        1 => true,
                        other => panic!("N = 1 but {} is not a bit!", other),
                    };
                }
            }
        } else {
            for (sub_corner, node) in subcoords(corner, cell.n - 1).into_iter().zip(cell.children) {
                self.raster_rec(sub_corner, image, rect, node);
            }
        }
    }
}

/// Calculates whether or not this square at `coord` of size `2^n` at time `2^(n - 1)` can be anything other than zero, given the input rect
fn zero_input(coord: Coord, n: usize, input_rect: Rect) -> bool {
    if n == 0 {
        return false;
    }

    let time = 1 << n - 1;
    let input_rect = extend_rect(input_rect, time);

    let width = 1 << n;
    let (x, y) = coord;
    !rect_intersect((coord, (x + width, y + width)), input_rect)
}

/// Check if the given point is inside the given rectangle
fn inside_rect((x, y): Coord, ((x1, y1), (x2, y2)): Rect) -> bool {
    debug_assert!(x1 < x2);
    debug_assert!(y1 < y2);
    x >= x1 && x < x2 && y >= y1 && y < y2
}

/// Calculate the row-major offset of the given `pos` inside the given `rect`, or return None if
/// out of bounds.
fn sample_rect(pos @ (x, y): Coord, rect @ ((x1, y1), (x2, _)): Rect) -> Option<usize> {
    inside_rect(pos, rect).then(|| {
        let (dx, dy) = (x - x1, y - y1);
        let width = x2 - x1;
        (dx + dy * width) as usize
    })
}

/// Given a corner position at `(x, y)`, and a size `n` return the corner positions of the four
/// quadrants that make up the macrocell.
fn subcoords((x, y): Coord, n: usize) -> [Coord; 4] {
    let side_len = 1 << n;
    [
        (x, y),
        (x + side_len, y),
        (x, y + side_len),
        (x + side_len, y + side_len),
    ]
}

/// Increase the width of the rect on all sides by `w`
fn extend_rect(((x1, y1), (x2, y2)): Rect, w: i64) -> Rect {
    ((x1 - w, y1 - w), (x2 + w, y2 + w))
}

/// Returns true if the given rectangles intersect
fn rect_intersect(a: Rect, b: Rect) -> bool {
    let ((x1a, y1a), (x2a, y2a)) = a;
    let ((x1b, y1b), (x2b, y2b)) = b;
    x1a < x2b && x1b < x2a && y1a < y2b && y1b < y2a
}

/// Solve a 4x4 grid
fn solve_4x4([[a, b, c, d], [e, f, g, h], [i, j, k, l], [m, n, o, p]]: [SubCells; 4]) -> SubCells {
    [
        solve_3x3([a, b, e, c, d, g, i, j, m]),
        solve_3x3([b, e, f, d, g, h, j, m, n]),
        solve_3x3([c, d, g, i, j, m, k, l, o]),
        solve_3x3([d, g, h, j, m, n, l, o, p]),
    ]
}

/// Solve a 3x4 grid
fn solve_3x3([a, b, c, d, e, f, g, h, i]: [Handle; 9]) -> Handle {
    let count = [a, b, c, d, f, g, h, i]
        .into_iter()
        .map(|Handle(i)| i)
        .sum();
    Handle(gol_rules(e.0 != 0, count) as usize)
}

/// Game of Life rules, otherwise known as B3/S23
fn gol_rules(center: bool, neighbors: usize) -> bool {
    match (center, neighbors) {
        (true, n) if (n == 2 || n == 3) => true,
        (false, n) if (n == 3) => true,
        _ => false,
    }
}

/// Returns the (width, height) of the given rect
pub fn rect_dimensions(((x1, y1), (x2, y2)): Rect) -> (i64, i64) {
    (x2 - x1, y2 - y1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_solve_4x4() {
        assert_eq!(
            solve_4x4([
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
            ]),
            [
                0, 1, //.
                1, 1, //.
            ]
            .map(Handle)
        );
    }

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
}
