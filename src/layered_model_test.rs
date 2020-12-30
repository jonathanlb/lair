use super::*;

use assert_approx_eq::assert_approx_eq;

use na::{Matrix1, Matrix1x3, Matrix2x1, Matrix2x3};
use na::{U1, U2, U3};

use crate::{LinearModel, Model};

#[test]
fn create_linear_model() {
    let mut model0 = LinearModel::<U3, U2>::new();
    let mut model1 = LinearModel::<U2, U1>::new();
    let model = LayeredModel::<U3, U2, U1> {
        model0: &mut model0,
        model1: &mut model1,
    };
    assert_eq!(model.num_inputs(), 3);
    assert_eq!(model.num_outputs(), 1);
}

#[test]
fn updates() {
    let mut model0 = LinearModel::<U3, U2>::new();
    let mut model1 = LinearModel::<U2, U1>::new();
    let model = LayeredModel::<U3, U2, U1> {
        model0: &mut model0,
        model1: &mut model1,
    };

    // XXX TODO
}
