use super::*;

use assert_approx_eq::assert_approx_eq;

use crate::{SGDTrainer, UpdateParams};
use na::{Matrix, Matrix1, Matrix1x2, Matrix1x3, Matrix2x1, Matrix2x3};
use na::{U1, U2};

const LEARNING_PARAMS: UpdateParams = UpdateParams {
    step_size: 0.001,
    l2_reg: 0.0,
};

#[test]
fn create_linear_model() {
    let mut trainer = SGDTrainer::new(&LEARNING_PARAMS);
    let model = LinearModel::<U2, U1>::new_random(&mut trainer);
    assert_eq!(model.num_inputs(), 2);
    assert_eq!(model.num_outputs(), 1);
}

#[test]
fn update_linear_model_improves_estimate() {
    let mut trainer = SGDTrainer::new(&LEARNING_PARAMS);
    let mut model = LinearModel::<U2, U1>::new_normal(&mut trainer, 100.0);

    let x0 = Matrix2x1::new(0.5, 1.0);
    let y0 = Matrix1x2::new(2.0, 1.0) * Matrix2x1::new(1.0, 1.0);
    model.update(&x0, &y0);
    let yh0 = model.predict(&x0);
    let e0 = Fxx::powf(y0[0] - yh0[0], 2.0);

    model.update(&x0, &y0);
    let yh1 = model.predict(&x0);
    let e1 = Fxx::powf(y0[0] - yh1[0], 2.0);

    assert!(
        e0 - e1 > 0.0,
        "failed to improve on update {} -> {}",
        e0,
        e1
    );
}

// This test seems like it should at least not degrade by approximation
// by more than some function of the step size and weights.....
#[test]
fn update_linear_model_improves_backprop_error() {
    let mut trainer = SGDTrainer::new(&LEARNING_PARAMS);
    let mut model = LinearModel::<U2, U1>::new_normal(&mut trainer, 100.0);
    println!("model state: w={} + b={}", model.ws, model.bs);

    fn f(x: &Matrix2x1<Fxx>) -> Fxx {
        2.0 * x[0] - 0.1 * x[1] + 1.0
    }

    let x0 = Matrix2x1::new(0.5, 1.0);
    let y0 = Matrix1::new(f(&x0));
    let de0 = model.update(&x0, &y0);
    println!("model state: w={} + b={}", model.ws, model.bs);

    let x1 = Matrix2x1::new(0.5, 1.0);
    let y1 = Matrix1::new(f(&x1));
    let de1 = model.update(&x1, &y1);
    println!("model state: w={} + b={}", model.ws, model.bs);

    let e0 = Matrix::norm(&de0);
    let e1 = Matrix::norm(&de1);
    assert!(
        e0 - e1 > -2.0 * LEARNING_PARAMS.step_size,
        "failed to improve on update {} -> {}",
        e0,
        e1
    );
}

#[test]
fn update_bulk_linear_model() {
    let mut trainer = SGDTrainer::new(&LEARNING_PARAMS);
    let mut model = LinearModel::<U2, U1>::new_random(&mut trainer);
    let x = Matrix2x3::new(2.0, 3.0, 4.0, 1.0, 4.0, 5.0);
    let y = Matrix1x3::new(6.0, 11.0, 14.0);
    assert_eq!(model.update_bulk(&x, &y), Ok(()));

    let x0 = Matrix2x1::new(0.5, 1.0);
    let yh = model.predict(&x0);
    assert_approx_eq!(yh[0], 3.0);
}

#[test]
fn update_bulk_unconstrained_linear_model() {
    let mut trainer = SGDTrainer::new(&LEARNING_PARAMS);
    let mut model = LinearModel::<U2, U1>::new_random(&mut trainer);
    let x = Matrix2x1::new(2.0, 1.0);
    let y = Matrix1::new(6.0);
    let updated = model.update_bulk(&x, &y);
    match updated {
        Ok(_) => panic!("update_bulk expected error from unconstrained update matrix"),
        Err(msg) => assert!(msg.starts_with("cannot update_bulk, no inverse for")),
    }
}
