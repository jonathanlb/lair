extern crate nalgebra as na;

use na::allocator::Allocator;
use na::storage::Owned;
use na::{DefaultAllocator, DimName, VectorN};

use crate::model::{Fxx, Model};

pub struct LayeredModel<'a, M: DimName, P: DimName, N: DimName> {
    model0: &'a mut dyn Model<M, P>,
    model1: &'a mut dyn Model<P, N>,
}

impl<'a, M, P, N> Model<M, N> for LayeredModel<'a, M, P, N>
where
    M: DimName,
    N: DimName,
    P: DimName,
    DefaultAllocator:
        Allocator<Fxx, N> + Allocator<Fxx, N, P> + Allocator<Fxx, P> + Allocator<Fxx, P, M>,
    Owned<Fxx, N>: Copy,
{
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
        let y0 = self.model0.predict(x);
        self.model1.predict(&y0)
    }

    fn update(&mut self, _x: &VectorN<Fxx, M>, _y: &VectorN<Fxx, N>) -> ()
    where
        DefaultAllocator: Allocator<Fxx, M> + Allocator<Fxx, N>,
    {
        // XXX TODO
    }
}

#[cfg(test)]
#[path = "./layered_model_test.rs"]
mod layered_model_test;
