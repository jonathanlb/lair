#[macro_use]
extern crate criterion;

mod conv2d;
mod linear;

criterion_main! {
   conv2d::conv2d,
   linear::linear,
}
