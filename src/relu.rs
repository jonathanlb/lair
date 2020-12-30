extern crate nalgebra as na;

use na::allocator::Allocator;
use na::{DefaultAllocator, DimName, VectorN};

use crate::model::{Fxx, Model};

pub struct Relu<'a, M: DimName, N: DimName> {
    model: &'a mut dyn Model<M, N>,
}

impl<'a, M, N> Model<M, N> for Relu<'a, M, N>
where
    M: DimName,
    N: DimName,
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
        let mut y = self.model.predict(x);
        for i in 0..self.num_outputs() {
            if y[i] < 0.0 {
                y[i] = 0.0;
            }
        }
        y
    }

    fn update(&mut self, x: &VectorN<Fxx, M>, y: &VectorN<Fxx, N>) -> ()
    where
        DefaultAllocator: Allocator<Fxx, M> + Allocator<Fxx, N>,
    {
        let mut yh = self.model.predict(x);
        for i in 0..self.num_outputs() {
            if !(yh[i] <= 0.0 && y[i] <= 0.0) {
                // correct the prediction if the underlying model didn't
                // predict correctly/a thresholded value
                yh[i] = y[i];
            }
        }
        self.model.update(x, &yh)
    }
}

#[cfg(test)]
#[path = "./relu_test.rs"]
mod relu_test;
