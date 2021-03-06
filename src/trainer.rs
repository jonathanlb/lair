extern crate nalgebra as na;

use crate::model::{has_nan, Fxx};
use log::debug;
use na::allocator::Allocator;
use na::storage::Owned;
use na::DefaultAllocator;
use na::{DimName, Matrix, MatrixMN, VectorN};

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

///
/// Just apply the gradient update at every iteration.
///
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
        debug_assert!(!has_nan(&ws_result), "NaN update weights");
        debug!(
            "update ({}x{}) |w|={} |b|={}",
            N::dim(),
            M::dim(),
            Matrix::norm(&ws_result),
            Matrix::norm(&bias_result)
        );
        Some((ws_result, bias_result))
    }
}

#[derive(Clone, Debug)]
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

///
/// Apply gradient updates at intervals.
///
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

pub struct MomentumTrainer<'a, M, N>
where
    M: DimName,
    N: DimName,
    DefaultAllocator: Allocator<Fxx, N, M> + Allocator<Fxx, N>,
{
    momentum: Fxx,
    velocity: Option<(MatrixMN<Fxx, N, M>, VectorN<Fxx, N>)>,
    gd: &'a mut dyn GradientTrainer<M, N>,
}

impl<'a, M, N> MomentumTrainer<'a, M, N>
where
    M: DimName,
    N: DimName,
    DefaultAllocator: Allocator<Fxx, N, M> + Allocator<Fxx, N>,
{
    pub fn new(momentum: Fxx, gd: &'a mut dyn GradientTrainer<M, N>) -> Self {
        MomentumTrainer {
            momentum: momentum,
            velocity: None,
            gd: gd,
        }
    }
}

///
/// Apply gradient a mixture of the most recent and next gradient updates.
/// Goodfellow, Bengio, Courville S8.3.2.
///
impl<'a, M, N> GradientTrainer<M, N> for MomentumTrainer<'a, M, N>
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
    ) -> Option<(MatrixMN<Fxx, N, M>, VectorN<Fxx, N>)> {
        match self.gd.train(weights, bias, gradient, bias_gradient) {
            Some((w1, b1)) => {
                match self.velocity {
                    Some((vw, vb)) => {
                        // gd trainer returns the updated value not the gradient
                        let gw = weights - w1;
                        let gb = bias - b1;
                        let vw1 = self.momentum * vw - gw;
                        let vb1 = self.momentum * vb - gb;
                        self.velocity = Some((vw1, vb1));
                        Some((weights + vw1, bias + vb1))
                    }
                    None => {
                        self.velocity = Some((w1 - weights, b1 - bias));
                        Some((w1, b1))
                    }
                }
            }
            None => None,
        }
    }
}

#[cfg(test)]
#[path = "./trainer_test.rs"]
mod trainer_test;
