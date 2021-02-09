use super::*;

use na::{U1, U2};

const UPDATE_PARAMS: UpdateParams = UpdateParams {
    l2_reg: 0.0,
    step_size: 0.001,
};

#[test]
fn sgd_trainer_updates() {
    let mut sgd = SGDTrainer::new(&UPDATE_PARAMS);

    let ws = MatrixMN::<Fxx, U1, U2>::new(0.0, 0.0);
    let b = VectorN::<Fxx, U1>::new(0.0);
    let gradient = MatrixMN::<Fxx, U1, U2>::new(1.0, 1.0);
    let bias_gradient = VectorN::<Fxx, U1>::new(1.0);

    match sgd.train(&ws, &b, &gradient, &bias_gradient) {
        Some((new_ws, new_bs)) => {
            // -1/2 * step_size
            assert_eq!(new_ws, MatrixMN::<Fxx, U1, U2>::new(-0.0005, -0.0005));
            assert_eq!(new_bs, VectorN::<Fxx, U1>::new(-0.0005));
        }
        None => assert!(false, "sgd trainer failed to update"),
    }
}

#[test]
fn batch_trainer_updates() {
    let mut bt = BatchTrainer::<U2, U1>::new(&UPDATE_PARAMS, 2);

    let ws = MatrixMN::<Fxx, U1, U2>::new(0.0, 0.0);
    let b = VectorN::<Fxx, U1>::new(0.0);
    let g0 = MatrixMN::<Fxx, U1, U2>::new(0.0, 1.0);
    let bg0 = VectorN::<Fxx, U1>::new(0.25);
    assert_eq!(bt.train(&ws, &b, &g0, &bg0), None);

    let g1 = MatrixMN::<Fxx, U1, U2>::new(1.0, 0.0);
    let bg1 = VectorN::<Fxx, U1>::new(0.75);
    assert_eq!(
        bt.train(&ws, &b, &g1, &bg1),
        Some((
            MatrixMN::<Fxx, U1, U2>::new(-0.00025, -0.00025),
            VectorN::<Fxx, U1>::new(-0.00025)
        ))
    );

    assert_eq!(bt.train(&ws, &b, &g0, &bg0), None);
}

#[test]
fn momentum_trainer_updates() {
    let mut sgd = SGDTrainer::new(&UPDATE_PARAMS);
    let mut mgd = MomentumTrainer::new(0.5, &mut sgd);

    let ws = MatrixMN::<Fxx, U1, U2>::new(0.0, 0.0);
    let b = VectorN::<Fxx, U1>::new(0.0);
    let gradient = MatrixMN::<Fxx, U1, U2>::new(1.0, 1.0);
    let bias_gradient = VectorN::<Fxx, U1>::new(1.0);

    let (new_ws, new_bs) = match mgd.train(&ws, &b, &gradient, &bias_gradient) {
        Some((new_ws, new_bs)) => {
            // -1/2 * step_size
            assert_eq!(new_ws, MatrixMN::<Fxx, U1, U2>::new(-0.0005, -0.0005));
            assert_eq!(new_bs, VectorN::<Fxx, U1>::new(-0.0005));
            (new_ws, new_bs)
        }
        None => {
            assert!(false, "sgd trainer failed to update");
            (ws, b) // not reached
        }
    };

    let gradient = MatrixMN::<Fxx, U1, U2>::new(-1.0, -1.0);
    let bias_gradient = VectorN::<Fxx, U1>::new(-1.0);

    match mgd.train(&new_ws, &new_bs, &gradient, &bias_gradient) {
        Some((ws1, bs1)) => {
            // slows down the steps
            assert_eq!(ws1, MatrixMN::<Fxx, U1, U2>::new(-0.00025, -0.00025));
            assert_eq!(bs1, VectorN::<Fxx, U1>::new(-0.00025));
        }
        None => assert!(false, "momentum trainer failed to update"),
    }
}
