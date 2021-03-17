extern crate nalgebra as na;

use log::debug;

use na::allocator::{Allocator, Reallocator};
use na::storage::Owned;
use na::DefaultAllocator;
use na::{DimAdd, DimName, DimSum, U1};
use na::{MatrixMN, VectorN};

use rand::distributions::{Distribution, Normal};

use crate::model::{has_nan, Fxx, Model};
use crate::trainer::GradientTrainer;

// #[derive(Clone, Copy, Debug)]
pub struct LinearModel<'a, M, N>
where
    M: DimName,
    N: DimName,
    DefaultAllocator: Allocator<Fxx, N>
        + Allocator<Fxx, N, M>
        + Allocator<Fxx, U1, M>
        + Allocator<Fxx, U1, N>
        + Allocator<Fxx, M, U1>
        + Allocator<Fxx, M, N>
        + Allocator<usize, M>
        + Allocator<usize, M, M>
        + Allocator<usize, N>
        + Allocator<usize, N, N>,
    Owned<Fxx, N>: Copy,
    Owned<Fxx, N, M>: Copy,
    Owned<usize, M>: Copy,
    Owned<usize, M, M>: Copy,
    Owned<usize, N>: Copy,
    Owned<usize, N, N>: Copy,
{
    trainer: &'a mut dyn GradientTrainer<M, N>,
    ws: MatrixMN<Fxx, N, M>,
    bs: VectorN<Fxx, N>,
}

impl<'a, M, N> Model<M, N> for LinearModel<'a, M, N>
where
    M: DimName,
    N: DimName,
    DefaultAllocator: Allocator<Fxx, N>
        + Allocator<Fxx, N, M>
        + Allocator<Fxx, U1, M>
        + Allocator<Fxx, U1, N>
        + Allocator<Fxx, M, U1>
        + Allocator<Fxx, M, N>
        + Allocator<usize, M>
        + Allocator<usize, M, M>
        + Allocator<usize, N>
        + Allocator<usize, N, N>,
    Owned<Fxx, N>: Copy,
    Owned<Fxx, N, M>: Copy,
    Owned<usize, M>: Copy,
    Owned<usize, M, M>: Copy,
    Owned<usize, N>: Copy,
    Owned<usize, N, N>: Copy,
{
    //
    // TODO: handle NaN trouble better.  There is also trouble printing ws in
    // implementing Mul for Copy trait
    //
    fn backpropagate(&mut self, x: &VectorN<Fxx, M>, de_dy: &VectorN<Fxx, N>) -> VectorN<Fxx, M> {
        debug!("linear backprop {}->{}", M::dim(), N::dim());
        debug_assert!(
            !has_nan(&x) && !has_nan(&de_dy),
            "backpropagate input error x={} de_dy={}",
            x,
            de_dy
        );
        debug_assert!(!has_nan(&self.ws), "backpropagate-3 NaN ws");
        let input_error = (de_dy.transpose() * self.ws).transpose();
        debug_assert!(
            !has_nan(&input_error),
            "backpropagate unstable return={} from {}T * w",
            input_error,
            de_dy
        ); // trouble printing self.ws

        let grad = de_dy * x.transpose();
        match self.trainer.train(&self.ws, &self.bs, &grad, de_dy) {
            Some((ws, bs)) => {
                self.ws = ws;
                self.bs = bs;
            }
            None => (),
        }
        input_error
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
        self.ws * x + self.bs
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

impl<'a, M, N> LinearModel<'a, M, N>
where
    M: DimName + DimAdd<U1>,
    N: DimName,
    DefaultAllocator: Allocator<Fxx, N>
        + Allocator<Fxx, N, M>
        + Allocator<Fxx, U1, M>
        + Allocator<Fxx, U1, N>
        + Allocator<Fxx, M, U1>
        + Allocator<Fxx, M, N>
        + Allocator<usize, M>
        + Allocator<usize, M, M>
        + Allocator<usize, N>
        + Allocator<usize, N, N>,
    Owned<Fxx, N>: Copy,
    Owned<Fxx, N, M>: Copy,
    Owned<usize, M>: Copy,
    Owned<usize, M, M>: Copy,
    Owned<usize, N>: Copy,
    Owned<usize, N, N>: Copy,
{
    pub fn merge(&mut self, a: Fxx, other: &LinearModel<M, N>) -> () {
        self.ws = (1.0 - a) * self.ws + a * other.ws;
        self.bs = (1.0 - a) * self.bs + a * other.bs;
    }

    pub fn new_normal(trainer: &'a mut dyn GradientTrainer<M, N>, std: Fxx) -> Self {
        let normal = Normal::new(0.0, std as f64);
        let mut rng = rand::thread_rng();

        macro_rules! rand {
            () => {
                |_r, _c| normal.sample(&mut rng) as Fxx
            };
        }
        let m = LinearModel {
            trainer: trainer,
            ws: MatrixMN::<Fxx, N, M>::from_fn(rand!()),
            bs: VectorN::<Fxx, N>::from_fn(rand!()),
        };
        m
    }

    pub fn new_random(trainer: &'a mut dyn GradientTrainer<M, N>) -> Self {
        let m = LinearModel {
            trainer: trainer,
            ws: MatrixMN::<Fxx, N, M>::new_random(),
            bs: VectorN::<Fxx, N>::new_random(),
        };
        m
    }

    pub fn update_bulk<D: DimName>(
        &mut self,
        x: &MatrixMN<Fxx, M, D>,
        y: &MatrixMN<Fxx, N, D>,
    ) -> Result<(), String>
    where
        DefaultAllocator: Reallocator<Fxx, M, D, DimSum<M, U1>, D>
            + Reallocator<Fxx, DimSum<M, U1>, D, D, DimSum<M, U1>>
            + Allocator<Fxx, D, M>
            + Allocator<Fxx, D, N>
            + Allocator<Fxx, N, D>
            + Allocator<Fxx, N, DimSum<M, U1>>
            + Allocator<Fxx, DimSum<M, U1>, DimSum<M, U1>>
            + Allocator<Fxx, M, D>
            + Allocator<usize, DimSum<M, U1>, DimSum<M, U1>>
            + Allocator<Fxx, M, M>,
        Owned<Fxx, M, D>: Copy,
        Owned<Fxx, D, DimSum<M, U1>>: Copy,
        Owned<Fxx, DimSum<M, U1>, D>: Copy,
        Owned<Fxx, DimSum<M, U1>, DimSum<M, U1>>: Copy,
    {
        let x1 = x.insert_row(self.num_inputs(), 1.0); // M+1 x D
        let x1t = x1.transpose(); // D x M+1
        let xxt = x1 * x1t; // M+1 x M+1

        if let Some(xxt1) = xxt.try_inverse() {
            let w1 = y * x1t * xxt1; // N x M+1, the last column represents b
            self.bs = w1.column(self.num_inputs()).into_owned();
            self.ws = MatrixMN::<Fxx, N, M>::from_column_slice(
                w1.columns(0, self.num_inputs()).into_owned().as_slice(),
            );
            Ok(())
        } else {
            let err = format!("cannot update_bulk, no inverse for {}", xxt);
            Err(err)
        }
    }

    pub fn get_ws(&self) -> &MatrixMN<Fxx, N, M> {
        &self.ws
    }
}

#[cfg(test)]
#[path = "./linear_model_test.rs"]
mod linear_model_test;
