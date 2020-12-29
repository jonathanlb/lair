use criterion::{criterion_group, criterion_main, Criterion};
use rand::distributions::{Distribution, Normal};

extern crate lair;
extern crate nalgebra as na;

use lair::{LinearModel, Model};
use na::{Matrix1x3, Matrix2x1, Matrix2x3};
use na::{U1, U2};

fn solve_simple_linear(sigma: f64) -> () {
    let normal = Normal::new(0.0, sigma);
    let mut model = LinearModel::<U2, U1>::new();
    let x = Matrix2x3::new(2.0, 3.0, 4.0, 1.0, 4.0, 5.0);

    let mut y = Matrix1x3::new(6.0, 11.0, 14.0);
    // Half of the benchmark time is spent here.
    y = Matrix1x3::from_iterator(
        y.iter()
            .map(|&x| x + normal.sample(&mut rand::thread_rng()) as f32),
    );

    assert_eq!(model.update_bulk(&x, &y), Ok(()));
    let x0 = Matrix2x1::new(0.5, 1.0);
    let _yh = model.predict(&x0);
}

fn solve_simple_linear_benchmark(c: &mut Criterion) {
    c.bench_function("solve_simple_linear", |b| {
        b.iter(|| solve_simple_linear(0.1))
    });
}

criterion_group!(benches, solve_simple_linear_benchmark);
criterion_main!(benches);
