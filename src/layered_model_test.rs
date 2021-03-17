use super::*;

use na::{Matrix, Matrix1, Vector3};
use na::{U1, U2, U3};

use rand::distributions::{Distribution, Normal};

use crate::{LinearModel, Model, SGDTrainer, setup_logging, UpdateParams};

const LEARNING_PARAMS: UpdateParams = UpdateParams {
    step_size: 1e-6,
    l2_reg: 0.0,
};

// Don't initialize for other tests -- it can only be done once...
#[cfg(test)]
#[ctor::ctor]
fn init() {
    setup_logging();
}

#[test]
fn create_layered_model() {
    let mut train0 = SGDTrainer::new(&LEARNING_PARAMS);
    let mut model0 = LinearModel::<U3, U2>::new_random(&mut train0);
    let mut train1 = SGDTrainer::new(&LEARNING_PARAMS);
    let mut model1 = LinearModel::<U2, U1>::new_random(&mut train1);
    let model = LayeredModel::<U3, U2, U1> {
        model0: &mut model0,
        model1: &mut model1,
    };
    assert_eq!(model.num_inputs(), 3);
    assert_eq!(model.num_outputs(), 1);
}

macro_rules! rand {
    ($n:ident, $r:ident) => {
        |_u, _c| $n.sample(&mut $r) as Fxx
    };
}

fn f(x: &Vector3<Fxx>) -> Matrix1<Fxx> {
    Matrix1::<Fxx>::new(2.0 * x[0] * x[0] - 3.0 * x[1] + x[1] * x[2] - 6.0)
}

#[test]
fn computes_gradient() {
    let mut train0 = SGDTrainer::new(&LEARNING_PARAMS);
    let mut model0 = LinearModel::<U3, U2>::new_normal(&mut train0, 10.0);
    let mut train1 = SGDTrainer::new(&LEARNING_PARAMS);
    let mut model1 = LinearModel::<U2, U1>::new_normal(&mut train1, 10.0);
    let mut model = LayeredModel::<U3, U2, U1> {
        model0: &mut model0,
        model1: &mut model1,
    };

    let normal = Normal::new(0.0, 5.0);
    let mut rng = rand::thread_rng();
    let x = Vector3::<Fxx>::from_fn(rand!(normal, rng));
    let y = f(&x);
    let yh = model.predict(&x);
    model.update(&x, &y);
    let yh1 = model.predict(&x);
    println!(
        "update backprop: x={} f(x)={} yh={} ... yh1={} ",
        x.transpose(),
        y,
        yh,
        yh1
    );
    let e0 = Matrix::norm(&(yh - y));
    let e1 = Matrix::norm(&(yh1 - y));
    println!("rms error {} -> {}", e0.sqrt(), e1.sqrt());
    assert!(e1 < e0);
}
