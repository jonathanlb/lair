use super::*;

use na::{Matrix, Matrix1, Vector3};
use na::{U1, U2, U3};

use rand::distributions::{Distribution, Normal};

use crate::{LinearModel, Model, UpdateParams};

const LEARNING_PARAMS: UpdateParams = UpdateParams { step_size: 0.01 };

#[test]
fn create_linear_model() {
    let mut model0 = LinearModel::<U3, U2>::new_random(&LEARNING_PARAMS);
    let mut model1 = LinearModel::<U2, U1>::new_random(&LEARNING_PARAMS);
    let model = LayeredModel::<U3, U2, U1> {
        model0: &mut model0,
        model1: &mut model1,
    };
    assert_eq!(model.num_inputs(), 3);
    assert_eq!(model.num_outputs(), 1);
}

#[test]
fn updates() {
    macro_rules! rand {
        ($n:ident, $r:ident) => {
            |_u, _c| $n.sample(&mut $r) as Fxx
        };
    }

    fn f(x: &Vector3<Fxx>) -> Matrix1<Fxx> {
        Matrix1::<Fxx>::new(2.0 * x[0] * x[0] - 3.0 * x[1] + x[1] * x[2] - 6.0)
    }

    fn eval_model(model: &dyn Model<U3, U1>, n: usize) -> Fxx {
        let normal = Normal::new(0.0, 100.0);
        let mut rng = rand::thread_rng();
        (0..n)
            .map(|_i| {
                let x = Vector3::<Fxx>::from_fn(rand!(normal, rng));
                Fxx::powf((model.predict(&x) - f(&x))[0], 2.0)
            })
            .sum::<Fxx>()
            / (n as Fxx)
    }

    let mut model0 = LinearModel::<U3, U2>::new_normal(&LEARNING_PARAMS, 100.0);
    let mut model1 = LinearModel::<U2, U1>::new_normal(&LEARNING_PARAMS, 100.0);
    let mut model = LayeredModel::<U3, U2, U1> {
        model0: &mut model0,
        model1: &mut model1,
    };

    let normal = Normal::new(0.0, 100.0);
    let mut rng = rand::thread_rng();
    let e0 = eval_model(&model, 10);
    for _i in 0..10 {
        let x = Vector3::<Fxx>::from_fn(rand!(normal, rng));
        let y = f(&x);
        let e = model.update(&x, &y);
        println!("update backprop: {} |{}|", e, Matrix::norm(&e));
        println!("mean sq error: {}", eval_model(&model, 10));
    }
    let e1 = eval_model(&model, 10);
    debug_assert!(e1 < e0);
}
