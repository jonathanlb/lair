extern crate nalgebra as na;

use lair::{Fxx, LayeredModel, LinearModel, Model, UpdateParams};
use na::{Matrix, Matrix1, Matrix2x1};
use na::{U1, U2};
use rand::seq::SliceRandom;
use rand::thread_rng;

fn f(x: &Matrix2x1<Fxx>) -> Matrix1<Fxx> {
    Matrix1::<Fxx>::new(0.5 * x[0] - 4.0 * x[1] - 6.0)
}

fn sample_input(sz: usize) -> Vec<Matrix2x1<Fxx>> {
    fn shuffled_xs(sz: usize) -> Vec<Fxx> {
        let mx = 0.5 * (sz as Fxx);
        let mut xs: Vec<Fxx> = (0..sz).map(|x| 0.1 * ((x as Fxx) - mx)).collect();
        xs.shuffle(&mut thread_rng());
        xs
    }

    shuffled_xs(sz)
        .iter()
        .zip(shuffled_xs(sz).iter())
        .map(|(&x0, &x1)| Matrix2x1::<Fxx>::new(x0, x1))
        .collect()
}

struct OptimizeParams {
    max_iter: usize,
    step_size: Fxx,
    test_batch: usize,
    train_batch: usize,
}

fn optimize_quadratic(params: &OptimizeParams) {
    let learning_rate = UpdateParams { step_size: params.step_size };

    let mut m0 = LinearModel::<U2, U2>::new_random(&learning_rate);
    let mut m1 = LinearModel::<U2, U1>::new_random(&learning_rate);
    let mut model = LayeredModel::<U2, U2, U1>::new(&mut m0, &mut m1);

    let mut i = 0;
    while i < params.max_iter {
        i += 1;
        let sample = sample_input(params.test_batch + params.train_batch);
        let (train, test) = sample.split_at(params.train_batch);
        for x in train {
            model.update(&x, &f(&x));
        }

        let err: Fxx = test
            .iter()
            .map(|x| Matrix::norm(&(f(x) - model.predict(x))))
            .sum::<Fxx>()
            / (params.test_batch as Fxx);
        println!("{} {}", i*params.train_batch, err);
    }
}

//
// Demonstration of network training to fit a quadratic.
//
fn main() {
    env_logger::init();
    let params = OptimizeParams {
        max_iter: 100,
        step_size: 1e-2,
        test_batch: 20,
        train_batch: 80,
    };
    println!("# step={} n_train={} n_test={}", params.step_size, params.train_batch, params.test_batch);
    optimize_quadratic(&params);
}
