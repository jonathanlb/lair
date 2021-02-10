use super::*;

use assert_approx_eq::assert_approx_eq;
use na::U2;
use std::fs::File;
use std::io::BufReader;

const BLACK: Fxx = 0.0;
const WHITE: Fxx = 65280.0;

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
