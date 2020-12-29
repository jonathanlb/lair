extern crate nalgebra as na;

pub type Fxx = f32;

use na::allocator::Allocator;
use na::DefaultAllocator;
use na::DimName;
use na::VectorN;

pub trait Model<M: DimName, N: DimName> {
    fn num_inputs(&self) -> usize;
    fn num_outputs(&self) -> usize;
    fn predict(&self, x: &VectorN<Fxx, M>) -> VectorN<Fxx, N>
    where
        DefaultAllocator: Allocator<Fxx, M> + Allocator<Fxx, N>;
    fn update(&mut self, x: &VectorN<Fxx, M>, y: &VectorN<Fxx, N>) -> ()
    where
        DefaultAllocator: Allocator<Fxx, M> + Allocator<Fxx, N>;
}
