use std::collections::HashMap;

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct Handle(pub(crate) usize);

type SubCells = [Handle; 4];

type Coord = (i64, i64);
type Rect = (Coord, Coord);

#[derive(Copy, Clone, Debug)]
struct MacroCell {
    /// The width of this cell is 2^n
    n: usize,
    /// Indices of child cells (each with width 2^(n-1))
    children: SubCells,
    /// The result of this macrocell 2^(n-2) time steps later, width width 2^(n-1).
    result: Option<Handle>,
}

pub struct HashLife {
    /// Mapping from sub-cells to parent cell
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
                    n: 0,
                    children: [Handle(0); 4],
                    result: Some(Handle(0)),
                },
                MacroCell {
                    n: 0,
                    children: [Handle(usize::MAX); 4],
                    result: Some(Handle(usize::MAX)),
                },
            ],
        }
    }

    /// Insert the given data with the given width and top-left
    /// corner and return a macrocell of width 2^n, padded with
    /// zeroes everywhere else. Top-left is relative to the top-left of the cell.
    pub fn insert_array(
        &mut self,
        input: &[bool],
        tl_corner: Coord,
        width: usize,
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
        let (top, left) = tl_corner;
        let br_corner = (top + height as i64, left + width as i64);
        let rect = (tl_corner, br_corner);

        self.insert_rect(input, tl_corner, rect, n)
    }

    pub fn insert_rect(
        &mut self,
        input: &[bool],
        tl_corner: Coord,
        rect: Rect,
        n: usize,
    ) -> Handle {
        // Short circuit for zeroes
        if zero_input(tl_corner, n, rect) {
            return Handle(0);
        }

        // Return the input pixel at the given coordinates
        if n == 0 {
            return Handle(
                sample_rect(tl_corner, rect)
                    .map(|idx| input[idx])
                    .unwrap_or(false) as usize,
            );
        }

        // Calculate which macrocell we are in
        let children = subcoords(tl_corner, n - 1)
            .map(|sub_corner| self.insert_rect(input, sub_corner, rect, n - 1));

        self.insert_cell(children, n - 1)
    }

    fn insert_cell(&mut self, children: SubCells, n: usize) -> Handle {
        match self.parent_cell.get(&children) {
            None => {
                let idx = self.macrocells.len();
                self.macrocells.push(MacroCell {
                    n,
                    children,
                    result: None,
                });
                let handle = Handle(idx);
                self.parent_cell.insert(children, handle);
                handle
            }
            Some(idx) => *idx,
        }
    }

    /// Returns the given macro`cell` advanced by the given number of `steps`, with
    pub fn result(&mut self, handle: Handle, steps: u128) -> Handle {
        let cell = self.macrocells[handle.0];
        assert!(
            cell.n >= 2,
            "Results can only be computed for 4x4 cells and larger"
        );

        // Check status of child cells
        let child_n = cell.n - 1;
        let skip_children = steps >> child_n & 1 == 0;

        let grandchild_n = child_n - 1;
        let skip_grandchildren = steps >> grandchild_n & 1 == 0;

        // Check if we already know the result
        // If we might skip one time step, we can't use the cached result because it's comprised of
        // two 2^(n-3) time steps
        if skip_children || skip_grandchildren {
            if let Some(result) = cell.result {
                return result;
            }
        }

        // Solve 4x4 if we're at n = 2 (so children are n - 1 = 1)
        // Skip if the relevant bit is set
        if child_n == 1 {
            self.insert_cell(
                match skip_grandchildren {
                    true => self.center_passthrough(handle),
                    false => solve_4x4(self.subcells(handle)),
                },
                child_n,
            )
        } else {
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
            ] = self.subcells(handle);

            //self.insert_cell()
            todo!()
        }
    }

    /// Get the center 4 cells of the given cell
    fn center_passthrough(&self, handle: Handle) -> SubCells {
        let [
            [_, _, _, d], // Top left
            [_, _, g, _], // Top right
            [_, j, _, _], // Bottom left
            [m, _, _, _] // Bottom right
        ] = self.subcells(handle);
        [d, g, j, m]
    }

    /// Children of the children of the given handle
    fn subcells(&self, Handle(parent): Handle) -> [SubCells; 4] {
        self.macrocells[parent]
            .children
            .map(|Handle(child)| self.macrocells[child].children)
    }
}

/// Calculates whether or not this square can be anything other than zero, given the input rect
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

/// Sample at `pos` from the given `input` buffer positioned at `rect`
fn sample_rect(pos @ (x, y): Coord, rect @ ((x1, y1), (x2, _)): Rect) -> Option<usize> {
    // debug_assert_eq!(input.len(), (x2 - x1) * (y2 - y1)) // TODO:
    inside_rect(pos, rect).then(|| {
        let (dx, dy) = (x - x1, y - y1);
        let width = x2 - x1; // Plus one since the rect is inclusive
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

/// Returns the (width, height) of the given rect
fn rect_dimensions(((x1, y1), (x2, y2)): Rect) -> (i64, i64) {
    (x2 - x1, y2 - y1)
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
