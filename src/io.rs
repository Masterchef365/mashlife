use anyhow::{bail, format_err, Result};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::iter::repeat;
use std::path::Path;

/// Import an RLE file
pub fn load_rle(path: impl AsRef<Path>) -> Result<(Vec<bool>, usize)> {
    let text = std::fs::read_to_string(path)?;
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

    'lines: for line in lines {
        for c in line.trim().chars() {
            match c {
                'b' | 'o' => {
                    let n = digits.parse().unwrap_or(1);
                    digits.clear();
                    x += n;
                    if x > width {}
                    data.extend(repeat(c == 'o').take(n));
                }
                '$' | '!' => {
                    match width.checked_sub(x) {
                        None => bail!("Pattern exceeds width!"),
                        Some(filler) => data.extend(repeat(false).take(filler)),
                    }
                    digits.clear();
                    x = 0;
                    if c == '!' {
                        break 'lines;
                    }
                }
                c if c.is_digit(10) => digits.push(c),
                c if c.is_whitespace() => (),
                _ => bail!("Unrecognized character {}", c),
            }
        }
    }

    let len = data.len();
    data.extend(repeat(false).take((width * height).checked_sub(len).unwrap()));

    /*if data.len() !=  {
        bail!(
            "Data length does not match header! {} vs {} * {} = {}",
            data.len(),
            width,
            height,
            width * height
        );
    }*/

    Ok((data, width))
}

pub fn write_png(path: impl AsRef<Path>, data: &[u8], width: usize) -> Result<()> {
    assert!(
        data.len() % 3 == 0,
        "Data length must be a multiple of 3 (RGB)"
    );
    let n_pixels = data.len() / 3;

    assert!(
        n_pixels % width == 0,
        "Pixel count must be a multiple of width"
    );
    let height = n_pixels / width;

    let file = File::create(path)?;
    let file = BufWriter::new(file);

    let mut encoder = png::Encoder::new(file, width as _, height as _); // Width is 2 pixels and height is 1.
    encoder.set_color(png::ColorType::Rgb);
    encoder.set_depth(png::BitDepth::Eight);
    let mut writer = encoder.write_header()?;

    writer.write_image_data(&data)?;

    Ok(())
}
