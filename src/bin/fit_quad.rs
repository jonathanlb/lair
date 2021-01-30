extern crate nalgebra as na;

use lair::{Fxx, LayeredModel, LinearModel, Model, UpdateParams};
use na::{Matrix, Matrix1, Matrix2x1};
use na::{U1, U2};
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
    #[structopt(short = "s", long = "step", default_value = "1e-3")]
    step_size: Fxx,
    #[structopt(short = "b", long = "test_size", default_value = "20")]
    test_batch: usize,
    #[structopt(short = "B", long = "train_batch", default_value = "80")]
    train_batch: usize,
}

fn f(x: &Matrix2x1<Fxx>) -> Matrix1<Fxx> {
    // Matrix1::<Fxx>::new(0.5 * x[0] - 4.0 * x[1] - 6.0)
    Matrix1::<Fxx>::new(
        0.5 * x[0] * x[0] 
        + 2.0 * x[0] * x[1] 
        - 4.0 * x[1] * x[1] 
        - 6.0)
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

fn optimize_quadratic(params: &OptimizeParams) {
    let learning_rate = UpdateParams {
        step_size: params.step_size,
    };

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
        println!("{} {}", i * params.train_batch, err);
    }
}

//
// Demonstration of network training to fit a quadratic.
//
fn main() {
    env_logger::init();
    let params = OptimizeParams::from_args();
    println!(
        "# step={} n_train={} n_test={}",
        params.step_size, params.train_batch, params.test_batch
    );
    optimize_quadratic(&params);
}
