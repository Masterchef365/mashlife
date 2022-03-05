pub mod io;
use std::collections::HashMap;
type ZwoHasher = std::hash::BuildHasherDefault<zwohash::ZwoHasher>;
use std::str::FromStr;

// TODO: This assumes you are only using one HashLife instance!!!
#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct Handle(pub(crate) usize);

pub type SubCells = [Handle; 4];

pub type Coord = (i64, i64);
pub type Rect = (Coord, Coord);

#[derive(Debug)]
pub struct MacroCell {
    /// The width of this cell is 2^n
    pub n: usize,
    /// Indices of child cells (each with width 2^(n-1))
    pub children: SubCells,
    /// The number of live cells in this macrocell
    pub alive: u32,
    /// Mapping from time step to result (if any)
    pub result: HashMap<usize, Handle, ZwoHasher>,
}

/// An implementation of HashLife;
/// Cellular-automata acceleration structure
pub struct HashLife {
    /// Mapping from (sub-cells and time step) to parent cell
    parent_cell: HashMap<SubCells, Handle, ZwoHasher>,
    /// Array of macrocells
    macrocells: Vec<MacroCell>,
    /// Ruleset
    rules: Rules,
}

impl HashLife {
    pub fn new(rules: Rules) -> Self {
        Self {
            rules,
            // Zero always yields zero
            parent_cell: [([Handle(0); 4], Handle(0))].into_iter().collect(),
            macrocells: vec![
                MacroCell {
                    n: 0,
                    alive: 0,
                    children: [Handle(0); 4],
                    result: Default::default(),
                },
                MacroCell {
                    n: 0,
                    alive: 1,
                    children: [Handle(usize::MAX); 4],
                    result: Default::default(),
                },
            ],
        }
    }

    pub fn macrocells(&self) -> impl Iterator<Item = (Handle, &MacroCell)> {
        self.macrocells
            .iter()
            .enumerate()
            .map(|(idx, cell)| (Handle(idx), cell))
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
            return Handle(
                sample_rect(tl_corner, input_rect)
                    .map(|idx| input[idx])
                    .unwrap_or(false) as usize,
            );
        }

        // Calculate which macrocell we are in
        let children = subcoords(tl_corner, n - 1)
            .map(|sub_corner| self.insert_rect(input, sub_corner, input_rect, n - 1));

        //self.insert_cell(children, n, Some((tl_corner, 0)))
        self.insert_cell(children, n)
    }

    /// Return the handle of the given cell, optionally setting it's creation coordinate (for
    /// visualization)
    fn insert_cell(
        &mut self,
        children: SubCells,
        n: usize,
    ) -> Handle {
        match self.parent_cell.get(&children) {
            None => {
                let idx = self.macrocells.len();
                let alive = children.iter().map(|&c| self.macrocell(c).alive).sum::<u32>();
                self.macrocells.push(MacroCell {
                    n,
                    children,
                    alive,
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
    pub fn result(&mut self, handle: Handle, dt: usize, corner: Coord, time: usize) -> Handle {
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

        assert!(dt <= 1 << cell.n - 2, "dt ({}) must be <= 2^(n - 2), n={}", dt, cell.n);

        // Check if we already know the result
        if let Some(&result) = cell.result.get(&dt) {
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
            self.insert_cell(result, cell_n - 1)
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
            .map(|(coord, subcells)| (coord, self.insert_cell(subcells, cell_n - 1)));

            /*
            Compute results or passthroughs for grandchild nodes
            | Q R S |
            | T U V |
            | W X Y |
            */

            let [q, r, s, t, u, v, w, x, y] =
                middle_3x3.map(|(coord, handle)| self.result(handle, dt_1, coord, time + 1));

            let corner_2x2 = |u, v| (grandchild_width + u * grandchild_width + cx, grandchild_width + v * grandchild_width + cy);

            // Get the middle four quadrants of the 3x3 above
            let middle_2x2 = [
                (corner_2x2(0, 0), [q, r, t, u]),
                (corner_2x2(1, 0), [r, s, u, v]),
                (corner_2x2(0, 1), [t, u, w, x]),
                (corner_2x2(1, 1), [u, v, x, y]),
            ]
            .map(|(coord, subcells)| (coord, self.insert_cell(subcells, cell_n - 1)));

            // Compute results or passthroughs for child nodes
            let result =
                middle_2x2.map(|(coord, handle)| self.result(handle, dt_2, coord, time + dt_1));

            // Save the result
            self.insert_cell(result, cell_n - 1)
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
        let mut image = vec![false; (width * height) as usize];

        let mut set_pixel = |pos: Coord, _| {
            if let Some(idx) = sample_rect(pos, rect) {
                image[idx] = true;
            }
        };

        self.resolve((0, 0), &mut set_pixel, 0, rect, root);

        image
    }

    /// Resolve the given handle into an image by calling the given function with pixel
    /// coordinates, and the number of live cells in the given cell.
    pub fn resolve(
        &self,
        corner: Coord,
        image: &mut impl FnMut(Coord, u32),
        min_n: usize,
        rect: Rect,
        handle: Handle,
    ) {
        // Check if the output is gauranteed to be zero (assumes input is zeroed!)
        if handle == Handle(0) {
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
            image(corner, cell.alive);
        } else {
            for (sub_corner, node) in subcoords(corner, cell.n - 1).into_iter().zip(cell.children) {
                self.resolve(sub_corner, image, min_n, rect, node);
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

/// Solve a 3x4 grid
fn solve_3x3([a, b, c, d, e, f, g, h, i]: [Handle; 9], rules: &Rules) -> Handle {
    let count = [a, b, c, d, f, g, h, i]
        .into_iter()
        .map(|Handle(i)| i)
        .sum();
    Handle(rules.execute(e.0 != 0, count) as usize)
}

#[derive(Copy, Clone, Debug)]
pub struct Rules {
    survive: [bool; 9],
    born: [bool; 9],
}

impl Rules {
    pub fn execute(&self, center: bool, neighbors: usize) -> bool {
        if center {
            self.survive[neighbors]
        } else {
            self.born[neighbors]
        }
    }
}

impl Default for Rules {
    fn default() -> Self {
        Self::from_str("B3/S23").unwrap()
    }
}

impl FromStr for Rules {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let mut parts = s.split('/');
        let born = parts
            .next()
            .ok_or_else(|| "No slash in rulestring".to_string())?
            .trim_start_matches('B');
        let survive = parts
            .next()
            .ok_or_else(|| "Empty rule".to_string())?
            .trim_start_matches('S');

        let to_rule_array = |s: &str| -> Result<[bool; 9], String> {
            let mut rules = [false; 9];
            for c in s.chars() {
                if c.is_digit(10) {
                    let dig = c as u8 - b'0';
                    let dig = dig as usize;
                    if dig < rules.len() {
                        rules[dig] = true;
                    } else {
                        return Err(format!("{dig} is not valid in a rule string"));
                    }
                }
            }
            Ok(rules)
        };

        Ok(Self {
            survive: to_rule_array(survive)?,
            born: to_rule_array(born)?,
        })
    }
}

/// Returns the (width, height) of the given rect
pub fn rect_dimensions(((x1, y1), (x2, y2)): Rect) -> (i64, i64) {
    (x2 - x1, y2 - y1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::str::FromStr;

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
