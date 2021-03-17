extern crate nalgebra as na;

use log::debug;

use na::allocator::Allocator;
use na::storage::Owned;
use na::Matrix;
use na::{DefaultAllocator, DimName, VectorN};

use crate::model::{has_nan, Fxx, Model};

pub struct LayeredModel<'a, M: DimName, P: DimName, N: DimName> {
    model0: &'a mut dyn Model<M, P>,
    model1: &'a mut dyn Model<P, N>,
}

impl<'a, M: DimName, P: DimName, N: DimName> LayeredModel<'a, M, P, N> {
    pub fn new(
        m0: &'a mut dyn Model<M, P>,
        m1: &'a mut dyn Model<P, N>,
    ) -> LayeredModel<'a, M, P, N> {
        LayeredModel {
            model0: m0,
            model1: m1,
        }
    }
}

impl<'a, M, P, N> Model<M, N> for LayeredModel<'a, M, P, N>
where
    M: DimName,
    N: DimName,
    P: DimName,
    DefaultAllocator: Allocator<Fxx, N>
        + Allocator<Fxx, N, P>
        + Allocator<Fxx, P>
        + Allocator<Fxx, P, M>
        + Allocator<usize, M>
        + Allocator<usize, N>
        + Allocator<usize, P>,
    Owned<Fxx, N>: Copy,
    Owned<usize, M>: Copy,
    Owned<usize, N>: Copy,
    Owned<usize, P>: Copy,
{
    fn backpropagate(&mut self, x: &VectorN<Fxx, M>, de_dy: &VectorN<Fxx, N>) -> VectorN<Fxx, M>
    where
        DefaultAllocator: Allocator<Fxx, N> + Allocator<Fxx, M>,
    {
        debug_assert!(
            !has_nan(&x) && !has_nan(&de_dy),
            "layered backpropagate input has nan"
        );

        let p = self.model0.predict(x);
        debug_assert!(
            !has_nan(&p),
            "layered intermediate result overflow m0({}) -> {}",
            x,
            p
        );

        let de_dp = self.model1.backpropagate(&p, de_dy);
        debug_assert!(
            !has_nan(&de_dp),
            "layered intermediate error overflow at m1({}) de_dy={} -> de_dp={}",
            p,
            de_dy,
            de_dp
        );
        debug!("|de_dp|={}", Matrix::norm(&de_dp));
        debug!("|x|={}", Matrix::norm(x));

        let de_dx = self.model0.backpropagate(x, &de_dp);
        debug!("|de_dx|={}", Matrix::norm(&de_dx));
        de_dx
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
        debug_assert!(!has_nan(&x), "invalid input {}", x);
        let y0 = self.model0.predict(x);
        debug_assert!(!has_nan(&y0), "invalid intermediate input {}", y0);
        self.model1.predict(&y0)
    }

    fn update(&mut self, x: &VectorN<Fxx, M>, y: &VectorN<Fxx, N>) -> VectorN<Fxx, M>
    where
        DefaultAllocator: Allocator<Fxx, M> + Allocator<Fxx, N>,
    {
        debug_assert!(!has_nan(&x) && !has_nan(&y), "layered update input has nan");

        let yh = self.predict(x);
        let err = yh - y;
        debug_assert!(
            !has_nan(&err) && !has_nan(&yh),
            "error overflow {} - {} = {}",
            yh,
            y,
            err
        );

        self.backpropagate(x, &err)
    }
}

#[cfg(test)]
#[path = "./layered_model_test.rs"]
mod layered_model_test;
