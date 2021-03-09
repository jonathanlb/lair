pub use self::layered_model::optimize_quadratic_benchmark;
pub use self::linear_model::solve_simple_linear_benchmark;

mod layered_model;
mod linear_model;

criterion_group!(
    linear,
    optimize_quadratic_benchmark,
    solve_simple_linear_benchmark
);
