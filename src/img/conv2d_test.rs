use super::*;

use na::{U1, U2, U3, U4, U6};

use crate::{LinearModel, SGDTrainer, UpdateParams};

const LEARNING_PARAMS: UpdateParams = UpdateParams {
    step_size: 1e-6,
    l2_reg: 0.0,
};

#[test]
fn create_conv_model() {
    let mut train0 = SGDTrainer::new(&LEARNING_PARAMS);
    let mut pooler = LinearModel::<U6, U1>::new_random(&mut train0);
    let mut cnn = Conv2d::<U2, U3, U1, U1, U3, U4>::new(&mut pooler);
}