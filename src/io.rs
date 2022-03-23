use anyhow::{bail, format_err, Result};
use std::iter::repeat;
use std::path::Path;

/// Load an RLE file
pub fn load_rle(path: impl AsRef<Path>) -> Result<(Vec<bool>, usize)> {
    parse_rle(&std::fs::read_to_string(path)?)
}

/// Parse RLE text
pub fn parse_rle(text: &str) -> Result<(Vec<bool>, usize)> {
    let mut lines = text.lines();

    // Find header
    let header_line = loop {
        let line = lines.next().ok_or(format_err!("Missing header"))?;
        if line.starts_with('#') {
            continue;
        } else {
            break line;
        }
    };

    // Parse header
    let header_err = || format_err!("Header failed to parse");
    let mut sections = header_line.split(',');

    let mut parse_section = |prefix: &str| -> Result<usize> {
        let sec = sections.next().ok_or_else(header_err)?;
        let mut halves = sec.split('=');

        let var_name = halves.next().ok_or_else(header_err)?.trim();
        let value = halves.next().ok_or_else(header_err)?.trim();

        if var_name != prefix {
            return Err(header_err());
        } else {
            Ok(value.parse()?)
        }
    };

    let (width, height) = (parse_section("x")?, parse_section("y")?);

    // Load data
    let mut data = vec![];

    let mut digits: String = "".into();
    let mut x = 0;

    let digits_or_one = |digits: &mut String| {
        let n = digits.parse().unwrap_or(1);
        digits.clear();
        n
    };

    'lines: for line in lines {
        for c in line.trim().chars() {
            match c {
                'b' | 'o' => {
                    let n = digits_or_one(&mut digits);
                    x += n;
                    data.extend(repeat(c == 'o').take(n));
                }
                '!' => break 'lines,
                '$' => {
                    let n = digits_or_one(&mut digits);

                    for _ in 0..n {
                        match width.checked_sub(x) {
                            None => bail!("Pattern exceeds width!"),
                            Some(filler) => data.extend(repeat(false).take(filler)),
                        }
                        x = 0;
                    }
                }
                c if c.is_digit(10) => digits.push(c),
                c if c.is_whitespace() => (),
                _ => bail!("Unrecognized character {}", c),
            }
        }
    }

    // Fill remaining zeroes
    let len = data.len();
    data.extend(repeat(false).take((width * height).checked_sub(len).unwrap()));

    Ok((data, width))
}
