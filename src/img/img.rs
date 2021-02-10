extern crate nalgebra as na;

use image::imageops::FilterType;
use image::io::Reader;
use image::ImageError;

use na::allocator::Allocator;
use na::DefaultAllocator;
use na::{DimName, MatrixMN};

use std::convert::TryInto;
use std::io::{BufRead, Seek};

use crate::model::Fxx;

///
/// Represent an image as a matrix, scaling it and converting it to greyscale.
///
pub fn read_luma<F, R, C>(reader: &mut F) -> Result<MatrixMN<Fxx, R, C>, ImageError>
where
    F: BufRead + Seek,
    R: DimName,
    C: DimName,
    DefaultAllocator: Allocator<Fxx, R, C>,
{
    let ir = Reader::new(reader)
        .with_guessed_format()
        .expect("Failed to read image");
    let image = ir.decode()?;
    let scaled = image
        .grayscale()
        .resize(
            C::dim().try_into().unwrap(),
            R::dim().try_into().unwrap(),
            FilterType::Nearest,
        )
        .to_luma16();
    Ok(MatrixMN::<Fxx, R, C>::from_fn(|r, c| {
        let p = *scaled.get_pixel(c as u32, r as u32);
        p[0] as Fxx
    }))
}

#[cfg(test)]
#[path = "./img_test.rs"]
mod img_test;
