//! Geometric types and calculations

/// Grid-space position
pub type Coord = (i64, i64);

/// Grid-space rectangle
pub type Rect = (Coord, Coord);

/// Check if the given point is inside the given rectangle
pub fn inside_rect((x, y): Coord, ((x1, y1), (x2, y2)): Rect) -> bool {
    debug_assert!(x1 < x2);
    debug_assert!(y1 < y2);
    x >= x1 && x < x2 && y >= y1 && y < y2
}

/// Calculate the row-major offset of the given `pos` inside the given `rect`, or return None if
/// out of bounds.
pub fn sample_rect(pos @ (x, y): Coord, rect @ ((x1, y1), (x2, _)): Rect) -> Option<usize> {
    inside_rect(pos, rect).then(|| {
        let (dx, dy) = (x - x1, y - y1);
        let width = x2 - x1;
        (dx + dy * width) as usize
    })
}

/// Given a corner position at `(x, y)`, and a size `n` return the corner positions of the four
/// quadrants that make up the macrocell.
pub fn subcoords((x, y): Coord, n: usize) -> [Coord; 4] {
    let side_len = 1 << n;
    [
        (x, y),
        (x + side_len, y),
        (x, y + side_len),
        (x + side_len, y + side_len),
    ]
}

/// Increase the width of the rect on all sides by `w`
pub fn extend_rect(((x1, y1), (x2, y2)): Rect, w: i64) -> Rect {
    ((x1 - w, y1 - w), (x2 + w, y2 + w))
}

/// Returns true if the given rectangles intersect
pub fn rect_intersect(a: Rect, b: Rect) -> bool {
    let ((x1a, y1a), (x2a, y2a)) = a;
    let ((x1b, y1b), (x2b, y2b)) = b;
    x1a < x2b && x1b < x2a && y1a < y2b && y1b < y2a
}

/// Returns the (width, height) of the given rect
pub fn rect_dimensions(((x1, y1), (x2, y2)): Rect) -> (i64, i64) {
    (x2 - x1, y2 - y1)
}

/// Return a rectangle with the given top-left corner and size 2^n
pub fn rect_pow_2(coord: Coord, n: usize) -> Rect {
    let width = 1 << n;
    let (x, y) = coord;
    (coord, (x + width, y + width))
}
