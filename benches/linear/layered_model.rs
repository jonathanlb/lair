use criterion::Criterion;
use log::debug;

extern crate nalgebra as na;

use lair::{Fxx, LayeredModel, LinearModel, Model, SGDTrainer, UpdateParams};

use na::allocator::Allocator;
use na::DefaultAllocator;
use na::{DimName, U1, U2};
use na::{Matrix, Matrix1, Matrix2x1, MatrixMN};

use rand::seq::SliceRandom;
use rand::thread_rng;

pub fn has_nan<M: DimName, N: DimName>(x: &MatrixMN<Fxx, M, N>) -> bool
where
    DefaultAllocator: Allocator<Fxx, M, N>,
{
    for i in 0..x.len() {
        let xi = x[i];
        if xi.is_nan() || xi.is_infinite() {
            return true;
        }
    }
    false
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

fn optimize_quadratic(learning_rate: &UpdateParams, tol: Fxx) {
    fn f(x: &Matrix2x1<Fxx>) -> Matrix1<Fxx> {
        Matrix1::<Fxx>::new(0.5 * x[0] - 4.0 * x[1] - 6.0)
    }

    let mut train0 = SGDTrainer::new(&learning_rate);
    let mut m0 = LinearModel::<U2, U2>::new_normal(&mut train0, 10.0);
    let mut train1 = SGDTrainer::new(&learning_rate);
    let mut m1 = LinearModel::<U2, U1>::new_normal(&mut train1, 10.0);
    let mut model = LayeredModel::<U2, U2, U1>::new(&mut m0, &mut m1);

    let num_samples = 100;
    let num_test = 20;
    let num_train = num_samples - num_test;
    loop {
        let sample = sample_input(num_samples);
        let (train, test) = sample.split_at(num_train);
        for x in train {
            debug_assert!(!has_nan(&x), "invalid input {}", x);
            let fx = f(&x);
            debug_assert!(!has_nan(&fx), "invalid output {}", fx);
            model.update(&x, &fx);
        }

        let me: Fxx = test
            .iter()
            .map(|x| {
                let fx = f(x);
                let y = model.predict(x);
                debug_assert!(!has_nan(&y), "NaN error {} -> {} ({})", x, y, fx);
                Matrix::norm(&(fx - y))
            })
            .sum::<Fxx>()
            / (num_test as Fxx);
        debug!("optimize_quadratic {} mean_error={}", me < tol, me);
        if me.is_nan() {
            panic!("NaN error");
        }
        if me < tol {
            break;
        }
    }
    debug!("complete optimize_quadratic");
}

// XXX Example of need to improve numeric stability in training.
// If step size is large we don't converge (diverge infact).
pub fn optimize_quadratic_benchmark(c: &mut Criterion) {
    env_logger::init();
    let learning_rate = UpdateParams {
        l2_reg: 0.0,
        step_size: 1e-5,
    };
    let tol = 0.1;

    c.bench_function("optimize_quadratic", |b| {
        b.iter(|| optimize_quadratic(&learning_rate, tol));
    });
}
