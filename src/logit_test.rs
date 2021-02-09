use super::*;

use assert_approx_eq::assert_approx_eq;

use na::{Matrix1, Matrix1x3, Matrix2x1, Matrix2x3};
use na::{U1, U2};

use crate::{LinearModel, Model, SGDTrainer, UpdateParams};

const LEARNING_PARAMS: UpdateParams = UpdateParams {
    step_size: 0.5,
    l2_reg: 0.0,
};

#[test]
fn applies_logit() {
    let mut trainer = SGDTrainer::new(&LEARNING_PARAMS);
    let mut underlying_model = LinearModel::<U2, U1>::new_random(&mut trainer);
    let x = Matrix2x3::new(2.0, 3.0, 4.0, 1.0, 4.0, 5.0);
    let y = Matrix1x3::new(6.0, 11.0, 14.0);
    assert_eq!(underlying_model.update_bulk(&x, &y), Ok(()));

    let model = Logit::<U2, U1>::new(&mut underlying_model);
    let x0 = Matrix2x1::new(0.5, 1.0);
    let y0 = model.predict(&x0);
    assert_approx_eq!(y0[0], 0.95257413); // logit(3.0)

    let x1 = Matrix2x1::new(-0.5, -1.0);
    let y1 = model.predict(&x1);
    assert_approx_eq!(y1[0], 0.26894142); // logit(-1.0)
}

#[test]
fn logit_updates() {
    let mut trainer = SGDTrainer::new(&LEARNING_PARAMS);
    let mut underlying_model = LinearModel::<U1, U1>::new_random(&mut trainer);
    let mut model = Logit::<U1, U1>::new(&mut underlying_model);

    let zero = Matrix1::new(0.0);
    let one = Matrix1::new(1.0);
    let two = Matrix1::new(2.0);

    model.update(&two, &one);
    model.update(&one, &zero);
    model.update(&zero, &zero);
    model.update(&two, &one);

    let y0 = model.predict(&two);
    model.update(&two, &one);
    let y1 = model.predict(&two);
    assert!(
        y1 > y0,
        "expect increased confidence, but {} -> {}",
        y0[0],
        y1[0]
    );
}
