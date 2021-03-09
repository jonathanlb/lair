extern crate nalgebra as na;

use image::imageops::FilterType;
use image::io::Reader;
use image::{ImageBuffer, ImageError, Luma};

use na::allocator::Allocator;
use na::DefaultAllocator;
use na::{DMatrix, DimName, MatrixMN, VectorN, U1};

use std::convert::TryInto;
use std::io::{BufRead, Seek};

use crate::model::Fxx;

const IMG_MAX: Fxx = 65535.0;

///
/// Scale the values of the input to span the visual range.
///
pub fn range_matrix<M, N>(mat: &MatrixMN<Fxx, M, N>) -> MatrixMN<Fxx, M, N>
where
    M: DimName,
    N: DimName,
    DefaultAllocator: Allocator<Fxx, M, N>,
{
    let max = mat.max();
    let min = mat.min();
    if max == min {
        MatrixMN::<Fxx, M, N>::from_element(0.5 * IMG_MAX)
    } else {
        let m = IMG_MAX / (mat.max() - mat.min());
        let b = 0.5 * ((IMG_MAX - m * max) - (m * min));
        MatrixMN::<Fxx, M, N>::from_fn(|i, j| m * mat[(i, j)] + b)
    }
}

///
/// Scale the values of the input to span the visual range.
///
pub fn range_vector<M>(v: &VectorN<Fxx, M>) -> VectorN<Fxx, M>
where
    M: DimName,
    DefaultAllocator: Allocator<Fxx, M>,
{
    range_matrix::<M, U1>(v)
}

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

pub fn read_lumad<F>(reader: &mut F, dim: (usize, usize)) -> Result<DMatrix<Fxx>, ImageError>
where
    F: BufRead + Seek,
{
    let num_rows = dim.0;
    let num_cols = dim.1;
    let ir = Reader::new(reader)
        .with_guessed_format()
        .expect("Failed to read image");
    let image = ir.decode()?;
    let scaled = image
        .grayscale()
        .resize(
            num_cols.try_into().unwrap(),
            num_rows.try_into().unwrap(),
            FilterType::Nearest,
        )
        .to_luma16();
    Ok(DMatrix::from_fn(num_rows, num_cols, |r, c| {
        let p = *scaled.get_pixel(c as u32, r as u32);
        p[0] as Fxx
    }))
}

pub fn write_luma_matrix<M, N>(data: &MatrixMN<Fxx, M, N>, path: &str) -> Result<(), ImageError>
where
    M: DimName,
    N: DimName,
    DefaultAllocator: Allocator<Fxx, M, N>,
{
    ImageBuffer::from_fn(N::dim() as u32, M::dim() as u32, |x, y| {
        Luma([data[(y as usize, x as usize)] as u16]) // XXX u16
    })
    .save(path)
}

pub fn write_luma_vector<M>(
    data: &VectorN<Fxx, M>,
    height: usize,
    width: usize,
    path: &str,
) -> Result<(), ImageError>
where
    M: DimName,
    DefaultAllocator: Allocator<Fxx, M>,
{
    debug_assert!(M::dim() == height * width);
    ImageBuffer::from_fn(width as u32, height as u32, |x, y| {
        Luma([data[(x + width as u32 * y) as usize] as u16]) // XXX u16....
    })
    .save(path)
}

#[cfg(test)]
#[path = "./img_test.rs"]
mod img_test;
