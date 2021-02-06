use super::*;

use assert_approx_eq::assert_approx_eq;

use na::{Matrix1, Matrix1x3, Matrix2x1, Matrix2x3};
use na::{U1, U2};

use crate::{LinearModel, Model, SGDTrainer, UpdateParams};

const LEARNING_PARAMS: UpdateParams = UpdateParams {
    step_size: 0.01,
    l2_reg: 0.0,
};

#[test]
fn create_linear_model() {
    let mut trainer = SGDTrainer::new(&LEARNING_PARAMS);
    let mut underlying_model = LinearModel::<U2, U1>::new_random(&mut trainer);
    let model = Relu::<U2, U1> {
        model: &mut underlying_model,
    };
    assert_eq!(model.num_inputs(), 2);
    assert_eq!(model.num_outputs(), 1);
}

#[test]
fn thresholds() {
    let mut trainer = SGDTrainer::new(&LEARNING_PARAMS);
    let mut underlying_model = LinearModel::<U2, U1>::new_random(&mut trainer);
    let x = Matrix2x3::new(2.0, 3.0, 4.0, 1.0, 4.0, 5.0);
    let y = Matrix1x3::new(6.0, 11.0, 14.0);
    assert_eq!(underlying_model.update_bulk(&x, &y), Ok(()));

    let model = Relu::<U2, U1> {
        model: &mut underlying_model,
    };
    let x0 = Matrix2x1::new(0.5, 1.0);
    let y0 = model.predict(&x0);
    assert_approx_eq!(y0[0], 3.0);

    let x1 = Matrix2x1::new(-0.5, -1.0);
    let y1 = model.predict(&x1);
    assert_approx_eq!(y1[0], 0.0);
}

#[test]
fn thresholds_update() {
    let mut trainer = SGDTrainer::new(&LEARNING_PARAMS);
    let mut underlying_model = LinearModel::<U2, U1>::new_random(&mut trainer);
    let x = Matrix2x3::new(2.0, 3.0, 4.0, 1.0, 4.0, 5.0);
    let y = Matrix1x3::new(6.0, 11.0, 14.0);
    assert_eq!(underlying_model.update_bulk(&x, &y), Ok(()));

    let mut model = Relu::<U2, U1> {
        model: &mut underlying_model,
    };

    // don't update contributions that should be thresholded
    let mut x0 = Matrix2x1::new(-1.0, -1.0);
    let mut y0 = Matrix1::new(0.0);
    model.update(&x0, &y0);
    x0 = Matrix2x1::new(0.5, 1.0);
    y0 = model.predict(&x0);
    assert_approx_eq!(y0[0], 3.0);

    // but update ones that should
    y0 = Matrix1::new(4.0);
    model.update(&x0, &y0);
    y0 = model.predict(&x0);
    assert!(y0[0] > 3.0);
}
