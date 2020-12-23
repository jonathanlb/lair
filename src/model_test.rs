use super::*;

use na::{Matrix1x2, Matrix2x1, Matrix2x3, Matrix1x3};
use na::{U1, U2};

#[test]
fn create_linear_model() {
    let model = LinearModel::<U2, U1>::new();
    assert_eq!(model.num_inputs(), 2);
    assert_eq!(model.num_outputs(), 1);
}

#[test]
fn update_linear_model() {
    let mut model = LinearModel::<U2, U1>::new();
    
    let x0 = Matrix2x1::new(0.5, 1.0);
    let y0 = Matrix1x2::new(2.0, 1.0) * Matrix2x1::new(1.0, 1.0);
    model.update(&x0, &y0);
    let yh0 = model.predict(&x0);
    let e0 = Fxx::powf(y0[0] - yh0[0], 2.0);

    model.update(&x0, &y0);
    let yh1 = model.predict(&x0);
    let e1 = Fxx::powf(y0[0] - yh1[0], 2.0);
     
    assert!(e0 - e1 > 0.0, "failed to improve on update {} -> {}", e0, e1);
}

#[test]
fn update_bulk_linear_model() {
    let mut model = LinearModel::<U2, U1>::new();
    let x = Matrix2x3::new(
        2.0, 3.0, 4.0,
        1.0, 4.0, 5.0);
    let y = Matrix1x3::new(6.0, 11.0, 14.0);
    model.update_bulk(&x, &y);
    
    let x0 = Matrix2x1::new(0.5, 1.0);
    let yh = model.predict(&x0);
    assert_eq!(yh[0], 3.0); // XXX approx
}
