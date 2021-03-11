extern crate nalgebra as na;

use image::imageops::FilterType;
use image::io::Reader;
use image::{ImageBuffer, ImageError, Luma};

use na::allocator::Allocator;
use nalgebra::storage::{Storage, StorageMut};
use na::DefaultAllocator;
use na::{DMatrix, DimName, Matrix, MatrixMN, Vector, VectorN, U1};

use std::convert::TryInto;
use std::io::{BufRead, Seek};
use log::debug;

use crate::model::Fxx;

const IMG_MAX: Fxx = 65535.0;

pub fn overlay_matrix<M, N, M1, N1, S, S1>(
    mat: &Matrix<Fxx, M, N, S>,
    dest_row: usize,
    dest_col: usize,
    dest: &mut Matrix<Fxx, M1, N1, S1>) -> () 
where
    M: DimName,
    N: DimName,
    M1: DimName,
    N1: DimName,
    S: Storage<Fxx, M, N>,
    S1: StorageMut<Fxx, M1, N1>,
{
    debug_assert!(mat.nrows() + dest_row <= dest.nrows(), "row overflow {} + {} > {}", dest_row, mat.nrows(), dest.nrows());
    debug_assert!(mat.ncols() + dest_col <= dest.ncols(), "column overflow {} + {} > {}", dest_col, mat.ncols(), dest.ncols());

    for col in 0..mat.ncols() {
        let col_off = col + dest_col;
        for row in 0..mat.nrows() {
            let row_off = row + dest_row;
            dest[(row_off, col_off)] = mat[(row, col)];
        }
    }
}

// nalgebra implements column-major matrices, we'll do the same.
pub fn overlay_matrix_to_vector<M, N, P, S, S1>(
    mat: &Matrix<Fxx, M, N, S>,
    dest_row: usize,
    dest_col: usize,
    dest: &mut Vector<Fxx, P, S1>,
    nrows: usize,
    ncols: usize) -> ()
where
    M: DimName,
    N: DimName,
    P: DimName,
    S: Storage<Fxx, M, N>,
    S1: StorageMut<Fxx, P>,
{
    debug_assert!(nrows*ncols == dest.nrows(), "expected {}x{} elements, but received {} element vector", nrows, ncols, dest.nrows());
    debug_assert!(mat.nrows() + dest_row <= nrows, "row overflow {} + {} > {}", dest_row, mat.nrows(), nrows);
    debug_assert!(mat.ncols() + dest_col <= ncols, "column overflow {} + {} > {}", dest_col, mat.ncols(), ncols);

    for col in 0..mat.ncols() {
        let col_off = (col + dest_col) * nrows;
        for row in 0..mat.nrows() {
            let row_off = row + dest_row;
            dest[row_off + col_off] = mat[(row, col)];
        }
    }
}

///
/// Scale the values of the input to span the visual range.
///
pub fn range_matrix<M, N, S>(mat: &Matrix<Fxx, M, N, S>) -> MatrixMN<Fxx, M, N>
where
    M: DimName,
    N: DimName,
    S: Storage<Fxx, M, N>,
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
pub fn range_vector<M, S>(v: &Vector<Fxx, M, S>) -> VectorN<Fxx, M>
where
    M: DimName,
    S: Storage<Fxx, M>,
    DefaultAllocator: Allocator<Fxx, M>,
{
    range_matrix::<M, U1, _>(v)
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
    debug!("copying {:?} image to {}x{} matrix", scaled.dimensions(), R::dim(), C::dim());
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

pub fn write_luma_matrix<M, N, S>(data: &Matrix<Fxx, M, N, S>, path: &str) -> Result<(), ImageError>
where
    M: DimName,
    N: DimName,
    S: Storage<Fxx, M, N>,
{
    ImageBuffer::from_fn(N::dim() as u32, M::dim() as u32, |x, y| {
        Luma([data[(y as usize, x as usize)] as u16]) // XXX u16
    })
    .save(path)
}

pub fn write_luma_vector<M, S>(
    data: &Vector<Fxx, M, S>,
    height: usize,
    width: usize,
    path: &str,
) -> Result<(), ImageError>
where
    M: DimName,
    S: Storage<Fxx, M>,
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
