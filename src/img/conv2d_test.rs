use super::*;

use na::{U1, U12, U2, U3, U4, U6};

use crate::{LinearModel, Model, SGDTrainer, UpdateParams};

const LEARNING_PARAMS: UpdateParams = UpdateParams {
    step_size: 1e-6,
    l2_reg: 0.0,
};

#[test]
fn create_conv_model() {
    let mut train0 = SGDTrainer::new(&LEARNING_PARAMS);
    let mut pooler = LinearModel::<U6, U1>::new_random(&mut train0);
    let mut cnn = Conv2d::<U2, U3, U1, U1, U3, U4, _, _>::new(&mut pooler);
    let x = VectorN::<Fxx, U12>::new_random();
    let y = VectorN::<Fxx, U4>::new_random();
    let _yh = cnn.predict(&x);
    cnn.update(&x, &y);
}

#[test]
fn extracts_input_patch() {
    let mut train0 = SGDTrainer::new(&LEARNING_PARAMS);
    let mut pooler = LinearModel::<U6, U1>::new_random(&mut train0);
    let cnn = Conv2d::<U2, U3, U1, U1, U3, U4, U12, U4>::new(&mut pooler);

    // let input = VectorN::<Fxx, U12>::from_vec(vec!(
    //    1.0, 2.0, 3.0, 4.0,
    //    5.0, 6.0, 7.0, 8.0,
    //    9.0, 10.0, 11.0, 12.0));
    let input = VectorN::<Fxx, U12>::from_fn(|i, _| i as Fxx + 1.0);
    let mut to_pool = cnn.get_input_patch(&input, 1, 1);
    let mut expected = VectorN::<Fxx, U6>::new(6.0, 7.0, 8.0, 10.0, 11.0, 12.0);
    assert_eq!(to_pool, expected);

    to_pool = cnn.get_input_patch(&input, 0, 1);
    expected = VectorN::<Fxx, U6>::new(2.0, 3.0, 4.0, 6.0, 7.0, 8.0);
    assert_eq!(to_pool, expected);

    to_pool = cnn.get_input_patch(&input, 1, 0);
    expected = VectorN::<Fxx, U6>::new(5.0, 6.0, 7.0, 9.0, 10.0, 11.0);
    assert_eq!(to_pool, expected);

    to_pool = cnn.get_input_patch(&input, 0, 0);
    expected = VectorN::<Fxx, U6>::new(1.0, 2.0, 3.0, 5.0, 6.0, 7.0);
    assert_eq!(to_pool, expected);
}

#[test]
fn extracts_output_patch() {
    let mut train0 = SGDTrainer::new(&LEARNING_PARAMS);
    let mut pooler = LinearModel::<U6, U1>::new_random(&mut train0);
    let cnn = Conv2d::<U2, U3, U1, U1, U3, U4, U12, U4>::new(&mut pooler);

    let output = VectorN::<Fxx, U4>::new(1.0, 2.0, 3.0, 4.0);
    let mut patch = cnn.get_output_error_patch(&output, 0, 0);
    assert_eq!(patch, VectorN::<Fxx, U1>::new(1.0));

    patch = cnn.get_output_error_patch(&output, 1, 0);
    assert_eq!(patch, VectorN::<Fxx, U1>::new(3.0));

    patch = cnn.get_output_error_patch(&output, 1, 1);
    assert_eq!(patch, VectorN::<Fxx, U1>::new(4.0));

    patch = cnn.get_output_error_patch(&output, 0, 1);
    assert_eq!(patch, VectorN::<Fxx, U1>::new(2.0));
}

#[test]
fn patches_output() {
    let mut train0 = SGDTrainer::new(&LEARNING_PARAMS);
    let mut pooler = LinearModel::<U6, U1>::new_random(&mut train0);
    let cnn = Conv2d::<U2, U3, U1, U1, U3, U4, U12, U4>::new(&mut pooler);

    let mut output = VectorN::<Fxx, U4>::zeros();
    let pooled = VectorN::<Fxx, U1>::new(1.0);
    cnn.patch_output(&pooled, &mut output, 0, 0);
    assert_eq!(output, VectorN::<Fxx, U4>::new(1.0, 0.0, 0.0, 0.0));

    cnn.patch_output(&pooled, &mut output, 1, 1);
    assert_eq!(output, VectorN::<Fxx, U4>::new(1.0, 0.0, 0.0, 1.0));

    cnn.patch_output(&pooled, &mut output, 1, 0);
    assert_eq!(output, VectorN::<Fxx, U4>::new(1.0, 0.0, 1.0, 1.0));

    cnn.patch_output(&pooled, &mut output, 0, 1);
    assert_eq!(output, VectorN::<Fxx, U4>::new(1.0, 1.0, 1.0, 1.0));
}

#[test]
fn patches_error() {
    // XXX
    let mut train0 = SGDTrainer::new(&LEARNING_PARAMS);
    let mut pooler = LinearModel::<U6, U1>::new_random(&mut train0);
    let cnn = Conv2d::<U2, U3, U1, U1, U3, U4, U12, U4>::new(&mut pooler);

    let mut pooled_error = VectorN::<Fxx, U12>::zeros();
    let err = VectorN::<Fxx, U6>::from_fn(|_, _| 1.0);
    let mut expected = VectorN::<Fxx, U12>::from_vec(vec![
        1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ]);
    cnn.patch_error(&err, &mut pooled_error, 0, 0);
    assert_eq!(pooled_error, expected);

    expected = VectorN::<Fxx, U12>::from_vec(vec![
        1.0, 1.0, 1.0, 0.0, 2.0, 2.0, 2.0, 0.0, 1.0, 1.0, 1.0, 0.0,
    ]);
    cnn.patch_error(&err, &mut pooled_error, 1, 0);
    assert_eq!(pooled_error, expected);

    expected = VectorN::<Fxx, U12>::from_vec(vec![
        1.0, 1.0, 1.0, 0.0, 2.0, 3.0, 3.0, 1.0, 1.0, 2.0, 2.0, 1.0,
    ]);
    cnn.patch_error(&err, &mut pooled_error, 1, 1);
    assert_eq!(pooled_error, expected);

    expected = VectorN::<Fxx, U12>::from_vec(vec![
        1.0, 2.0, 2.0, 1.0, 2.0, 4.0, 4.0, 2.0, 1.0, 2.0, 2.0, 1.0,
    ]);
    cnn.patch_error(&err, &mut pooled_error, 0, 1);
    assert_eq!(pooled_error, expected);
}
