extern crate nalgebra as na;

pub type Fxx = f32;

use na::allocator::Allocator;
use nalgebra::storage::Storage;
use na::DefaultAllocator;
use na::DimName;
use na::{Matrix, VectorN};

pub trait Model<M: DimName, N: DimName> {
    /// Apply backpropagation to this layer/module of a neural network,
    /// returning the backpropagated error for the layer/module creating
    /// input for this layer/module.
    ///
    /// # Arguments
    ///
    /// * `x` - the input at which the model is being trained.
    /// * `de_dy` - the error partial derivative with respect to the output of
    /// this model.
    fn backpropagate(&mut self, x: &VectorN<Fxx, M>, de_dy: &VectorN<Fxx, N>) -> VectorN<Fxx, M>
    where
        DefaultAllocator: Allocator<Fxx, N> + Allocator<Fxx, M>;

    fn num_inputs(&self) -> usize;
    fn num_outputs(&self) -> usize;

    /// Run the model to predict a value for the input x.
    ///
    /// # Arguments
    ///
    /// * `x` - input for which to compute a modeled value.
    fn predict(&self, x: &VectorN<Fxx, M>) -> VectorN<Fxx, N>
    where
        DefaultAllocator: Allocator<Fxx, M> + Allocator<Fxx, N>;

    /// Update a model with an observation, y, from given input, x, returning
    /// the gradient of the input to be used for backpropogation.
    ///
    /// # Arguments
    /// * `x` - input corresponding to the observation y.
    /// * `y` - observed/"correct" value corresponding to the input x.
    fn update(&mut self, x: &VectorN<Fxx, M>, y: &VectorN<Fxx, N>) -> VectorN<Fxx, M>
    where
        DefaultAllocator: Allocator<Fxx, M> + Allocator<Fxx, N>;
}

pub fn has_nan<M, N, S>(x: &Matrix<Fxx, M, N, S>) -> bool
where
    M: DimName,
    N: DimName,
    S: Storage<Fxx, M, N>,
{
    for i in 0..x.len() {
        let xi = x[i];
        if xi.is_nan() || xi.is_infinite() {
            return true;
        }
    }
    false
}
