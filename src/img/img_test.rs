use super::*;

use assert_approx_eq::assert_approx_eq;
use na::{DimProd, U1, U2, U3, U4};
use std::fs::File;
use std::io::BufReader;

const BLACK: Fxx = 0.0;
const WHITE: Fxx = 65280.0;

#[test]
fn overlay_matrix_simple() {
    let mut dest = MatrixMN::<Fxx, U3, U4>::zeros();
    let mat = MatrixMN::<Fxx, U2, U2>::from_element(1.0);

    overlay_matrix(&mat, 1, 1, &mut dest);
    assert_eq!(dest[(1,1)], 1.0);
    assert_eq!(dest[(1,2)], 1.0);
    assert_eq!(dest[(2,1)], 1.0);
    assert_eq!(dest[(2,2)], 1.0);
}

#[test]
fn overlay_matrix_to_vector_simple() {
    let mut dest = VectorN::<Fxx, DimProd<U3, U4>>::zeros();
    let mat = MatrixMN::<Fxx, U1, U2>::from_element(1.0);

    overlay_matrix_to_vector(&mat, 1, 2, &mut dest, U3::dim(), U4::dim());
    let expected = MatrixMN::<Fxx, U3, U4>::new(
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 1.0,
        0.0, 0.0, 0.0, 0.0);
    assert_eq!(dest.data.as_slice(), expected.data.as_slice());
}

#[test]
fn read_luma_image() {
    let f = File::open("./test-data/img/test-checker.png").unwrap();
    let mut br = BufReader::new(f);
    match read_luma::<BufReader<File>, U2, U2>(&mut br) {
        Ok(mat) => {
            assert_eq!(mat.nrows(), 2);
            assert_eq!(mat.ncols(), 2);
            println!("image read: {}", mat);
            assert_approx_eq!(mat[0], BLACK);
            assert_approx_eq!(mat[1], WHITE);
            assert_approx_eq!(mat[2], WHITE);
            assert_approx_eq!(mat[3], BLACK);
        }
        Err(e) => {
            panic!("failed to read test image: {}", e)
        }
    }
}
