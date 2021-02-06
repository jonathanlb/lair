extern crate nalgebra as na;

use lair::{
    BatchTrainer, Fxx, GradientTrainer, LayeredModel, LinearModel, Model, SGDTrainer, UpdateParams,
};
use log::debug;
use na::{Matrix, Matrix1, Matrix2x1};
use na::{U1, U2};
use rand::distributions::{Distribution, Uniform};
use rand::seq::SliceRandom;
use rand::thread_rng;
use structopt::StructOpt;

#[derive(Debug, StructOpt)]
#[structopt(
    name = "fit quadratic example",
    about = "Demonstrate batch convergence training on a simple function."
)]
struct OptimizeParams {
    #[structopt(short = "n", long = "iter", default_value = "100")]
    max_iter: usize,
    #[structopt(short = "m", long = "mini-batch", default_value = "0")]
    mini_batch: usize,
    #[structopt(short = "s", long = "step", default_value = "1e-3")]
    step_size: Fxx,
    #[structopt(short = "b", long = "test_size", default_value = "20")]
    test_batch: usize,
    #[structopt(short = "B", long = "train_batch", default_value = "80")]
    train_batch: usize,
    #[structopt(short = "l", long = "l2", default_value = "0.0")]
    l2: Fxx,
}

fn f(x: &Matrix2x1<Fxx>) -> Matrix1<Fxx> {
    // Matrix1::<Fxx>::new(0.5 * x[0] - 4.0 * x[1] - 6.0)
    Matrix1::<Fxx>::new(0.5 * x[0] * x[0] + 2.0 * x[0] * x[1] - 4.0 * x[1] * x[1] - 6.0)
}

fn sample_input(sz: usize) -> Vec<Matrix2x1<Fxx>> {
    fn shuffled_xs(sz: usize) -> Vec<Fxx> {
        const MX_ABS: Fxx = 10.0;
        let mut rng = thread_rng();
        let dist = Uniform::from(-MX_ABS..MX_ABS);
        let mut xs: Vec<Fxx> = (0..sz).map(|_| dist.sample(&mut rng)).collect();
        xs.shuffle(&mut rng);
        xs
    }

    shuffled_xs(sz)
        .iter()
        .zip(shuffled_xs(sz).iter())
        .map(|(&x0, &x1)| Matrix2x1::<Fxx>::new(x0, x1))
        .collect()
}

fn optimize_quadratic(params: &OptimizeParams) {
    let learning_rate = UpdateParams {
        step_size: params.step_size,
        l2_reg: params.l2,
    };
    let mut train0: Box<dyn GradientTrainer<U2, U2>> = if params.mini_batch > 0 {
        Box::new(BatchTrainer::<U2, U2>::new(
            &learning_rate,
            params.mini_batch,
        ))
    } else {
        Box::new(SGDTrainer::new(&learning_rate))
    };
    let mut m0 = LinearModel::<U2, U2>::new_random(&mut *train0);

    let mut train1: Box<dyn GradientTrainer<U2, U1>> = if params.mini_batch > 0 {
        Box::new(BatchTrainer::<U2, U1>::new(
            &learning_rate,
            params.mini_batch,
        ))
    } else {
        Box::new(SGDTrainer::new(&learning_rate))
    };
    let mut m1 = LinearModel::<U2, U1>::new_random(&mut *train1);
    let mut model = LayeredModel::<U2, U2, U1>::new(&mut m0, &mut m1);

    let mut i = 0;
    while i < params.max_iter {
        i += 1;
        let sample = sample_input(params.test_batch + params.train_batch);
        let (train, test) = sample.split_at(params.train_batch);
        for x in train {
            let y = f(&x);
            let yh = model.predict(&x);
            let e = model.update(&x, &y);
            let yh1 = model.predict(&x);
            debug!(
                "({},{}) -> {}/{} e={} de={} delta={}",
                x[0],
                x[1],
                y[0],
                yh[0],
                Matrix::norm(&(yh - y)),
                Matrix::norm(&e),
                Matrix::norm(&(yh - y)) - Matrix::norm(&(yh1 - y)),
            );
        }

        let error_sums = test
            .iter()
            .map(|x| {
                let y = f(x);
                let yh = model.predict(x);
                let n = Matrix::norm(&(yh - y));
                (n, n * n)
            })
            .fold((0.0, 0.0), |sum, i| (sum.0 + i.0, sum.1 + i.1));

        let mean_error = error_sums.0 / (params.test_batch as Fxx);
        let mean2_error = error_sums.1 / (params.test_batch as Fxx);
        println!(
            "{} {} {}",
            i * params.train_batch,
            mean_error,
            (mean2_error - mean_error.powi(2)).sqrt()
        );
    }
}

//
// Demonstration of network training to fit a quadratic.
//
fn main() {
    env_logger::init();
    let params = OptimizeParams::from_args();
    println!(
        "# step={} n_train={} n_test={} mini_batch={}",
        params.step_size, params.train_batch, params.test_batch, params.mini_batch
    );
    optimize_quadratic(&params);
}
