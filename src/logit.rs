extern crate nalgebra as na;

use log::debug;

use na::allocator::Allocator;
use na::{DefaultAllocator, DimName, VectorN};

use crate::model::{has_nan, Fxx, Model};

pub struct Logit<'a, M: DimName, N: DimName> {
    model: &'a mut dyn Model<M, N>,
}

/// The logistic function evaluated at x.
fn logit(x: Fxx) -> Fxx {
    1.0 / (1.0 + Fxx::exp(-x))
}

/// Derivative  of the logistic with respect to x at x.
fn dlogit(x: Fxx) -> Fxx {
    // Fxx::exp(x) / (1.0 + Fxx::exp(x)).powi(2); // unstable
    logit(x) * logit(-x)
}

impl<'a, M, N> Logit<'a, M, N>
where
    M: DimName,
    N: DimName,
{
    pub fn new(model: &'a mut dyn Model<M, N>) -> Self {
        Logit { model: model }
    }
}

impl<'a, M, N> Model<M, N> for Logit<'a, M, N>
where
    M: DimName,
    N: DimName,
{
    fn backpropagate(&mut self, x: &VectorN<Fxx, M>, de_dy: &VectorN<Fxx, N>) -> VectorN<Fxx, M>
    where
        DefaultAllocator: Allocator<Fxx, M> + Allocator<Fxx, N>,
    {
        debug!("logit backprop {}->{}", M::dim(), N::dim());
        debug_assert!(!has_nan(&x), "backprop x has_nan");
        debug_assert!(!has_nan(&de_dy), "backprop de_dy has_nan");

        let p = self.model.predict(x);
        debug_assert!(!has_nan(&p), "backprop p has_nan");
        
        let de_dp = VectorN::<Fxx, N>::from_fn(|r, _c| dlogit(p[r]) * de_dy[r]);
        debug_assert!(!has_nan(&de_dp), "backprop de_dp has_nan");
        self.model.backpropagate(x, &de_dp)
    }

    #[inline]
    fn num_inputs(&self) -> usize {
        M::dim()
    }

    #[inline]
    fn num_outputs(&self) -> usize {
        N::dim()
    }

    fn predict(&self, x: &VectorN<Fxx, M>) -> VectorN<Fxx, N>
    where
        DefaultAllocator: Allocator<Fxx, M> + Allocator<Fxx, N>,
    {
        let mut y = self.model.predict(x);
        for i in 0..self.num_outputs() {
            y[i] = logit(y[i])
        }
        debug_assert!(!has_nan(&y), "logit predict has_nan");
        y
    }

    fn update(&mut self, x: &VectorN<Fxx, M>, y: &VectorN<Fxx, N>) -> VectorN<Fxx, M>
    where
        DefaultAllocator: Allocator<Fxx, M> + Allocator<Fxx, N>,
    {
        let yh = self.predict(x);
        let err = yh - y;
        self.backpropagate(x, &err)
    }
}

#[cfg(test)]
#[path = "./logit_test.rs"]
mod logit_test;
