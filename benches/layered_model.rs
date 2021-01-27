use criterion::{criterion_group, criterion_main, Criterion};

extern crate nalgebra as na;

use lair::{Fxx, LayeredModel, LinearModel, Model, UpdateParams};

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

    let mut m0 = LinearModel::<U2, U2>::new_random(&learning_rate);
    let mut m1 = LinearModel::<U2, U1>::new_random(&learning_rate);
    let mut model = LayeredModel::<U2, U2, U1>::new(&mut m0, &mut m1);

    let tol2 = Fxx::powf(tol, 2.0);
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

        let msq: Fxx = test
            .iter()
            .map(|x| {
                let fx = f(x);
                let y = model.predict(x);
                debug_assert!(!has_nan(&y), "NaN error {} -> {} ({})", x, y, fx);
                Matrix::norm(&(fx - y))
            })
            .sum::<Fxx>()
            / (num_test as Fxx);
        println!("mean_error: {}", msq);
        if msq.is_nan() {
            panic!("NaN error");
        }
        if msq < tol2 {
            break;
        }
    }
}

fn optimize_quadratic_benchmark(c: &mut Criterion) {
    let learning_rate = UpdateParams { step_size: 0.01 };
    let tol = 0.1;

    c.bench_function("optimize_quadratic", |b| {
        b.iter(|| optimize_quadratic(&learning_rate, tol));
    });
}

criterion_group!(benches, optimize_quadratic_benchmark);
criterion_main!(benches);
