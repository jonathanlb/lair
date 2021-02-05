extern crate nalgebra as na;

use crate::model::{has_nan, Fxx};
use na::allocator::Allocator;
use na::storage::Owned;
use na::DefaultAllocator;
use na::{DimName, MatrixMN, VectorN};

#[derive(Clone, Copy, Debug)]
pub struct UpdateParams {
    pub l2_reg: Fxx,
    pub step_size: Fxx,
}

//
// GradientTrainer is an interface to record loss-function gradients
// from training observations for model update.
//
// # Arguments
// * `M` the number of scalar inputs to the model.
// * `N` the number of scalar outputs from the model.
pub trait GradientTrainer<M: DimName, N: DimName> {
    fn train(
        &mut self,
        weights: &MatrixMN<Fxx, N, M>,
        bias: &VectorN<Fxx, N>,
        gradient: &MatrixMN<Fxx, N, M>,
        bias_gradient: &VectorN<Fxx, N>,
    ) -> Option<(MatrixMN<Fxx, N, M>, VectorN<Fxx, N>)>
    where
        DefaultAllocator: Allocator<Fxx, N, M> + Allocator<Fxx, N>;
}

#[derive(Clone, Copy, Debug)]
pub struct SGDTrainer<'a> {
    update_params: &'a UpdateParams,
}

impl<'a> SGDTrainer<'a> {
    pub fn new(update_params: &'a UpdateParams) -> Self {
        SGDTrainer {
            update_params: update_params,
        }
    }
}

impl<'a, M, N> GradientTrainer<M, N> for SGDTrainer<'a>
where
    M: DimName,
    N: DimName,
{
    fn train(
        &mut self,
        weights: &MatrixMN<Fxx, N, M>,
        bias: &VectorN<Fxx, N>,
        gradient: &MatrixMN<Fxx, N, M>,
        bias_gradient: &VectorN<Fxx, N>,
    ) -> Option<(MatrixMN<Fxx, N, M>, VectorN<Fxx, N>)>
    where
        DefaultAllocator: Allocator<Fxx, N, M> + Allocator<Fxx, N>,
    {
        let step_size = self.update_params.step_size / (M::dim() as Fxx);
        let bias_result = bias - step_size * bias_gradient;
        let ws_result = (1.0 - self.update_params.l2_reg) * weights - step_size * gradient;
        // trouble printing weights
        debug_assert!(!has_nan(&ws_result), "NaN update weights");
        Some((ws_result, bias_result))
    }
}

// #[derive(Clone, Copy, Debug)]
pub struct BatchTrainer<'a, M, N>
where
    M: DimName,
    N: DimName,
    DefaultAllocator: Allocator<Fxx, N, M> + Allocator<Fxx, N>,
{
    batch_num: usize,
    batch_size: usize,
    grads: Vec<(MatrixMN<Fxx, N, M>, VectorN<Fxx, N>)>,
    sgd: SGDTrainer<'a>,
}

impl<'a, M, N> BatchTrainer<'a, M, N>
where
    M: DimName,
    N: DimName,
    DefaultAllocator: Allocator<Fxx, N, M> + Allocator<Fxx, N>,
{
    pub fn new(update_params: &'a UpdateParams, batch_size: usize) -> Self {
        let default_pair = (MatrixMN::<Fxx, N, M>::zeros(), VectorN::<Fxx, N>::zeros());
        BatchTrainer {
            batch_num: 0,
            batch_size: batch_size,
            grads: vec![default_pair; batch_size],
            sgd: SGDTrainer::new(update_params),
        }
    }
}

impl<'a, M, N> GradientTrainer<M, N> for BatchTrainer<'a, M, N>
where
    M: DimName,
    N: DimName,
    DefaultAllocator: Allocator<Fxx, N, M> + Allocator<Fxx, N>,
    Owned<Fxx, N>: Copy,
    Owned<Fxx, N, M>: Copy,
{
    fn train(
        &mut self,
        weights: &MatrixMN<Fxx, N, M>,
        bias: &VectorN<Fxx, N>,
        gradient: &MatrixMN<Fxx, N, M>,
        bias_gradient: &VectorN<Fxx, N>,
    ) -> Option<(MatrixMN<Fxx, N, M>, VectorN<Fxx, N>)>
    where
        DefaultAllocator: Allocator<Fxx, N, M> + Allocator<Fxx, N>,
    {
        self.grads[self.batch_num] = (*gradient, *bias_gradient);
        self.batch_num += 1;
        if self.batch_num >= self.batch_size {
            self.batch_num = 0;
            let (mut g_sum, mut bg_sum) = self.grads[1..]
                .iter()
                .fold(self.grads[0], |sum, pair| (sum.0 + pair.0, sum.1 + pair.1));
            let n1 = 1.0 / self.batch_size as Fxx;
            g_sum *= n1;
            bg_sum *= n1;
            self.sgd.train(weights, bias, &g_sum, &bg_sum)
        } else {
            None
        }
    }
}
#[cfg(test)]
#[path = "./trainer_test.rs"]
mod trainer_test;
