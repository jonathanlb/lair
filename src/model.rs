extern crate nalgebra as na;

type Fxx = f32;

use na::{MatrixMN, VectorN };
use na::{DimName, DimAdd, DimSum, U1};
use na::{DefaultAllocator};
use na::allocator::{Allocator, Reallocator};
use na::storage::Owned;

pub trait Model<M, N> where 
    M: DimName,
    N: DimName,
    DefaultAllocator: Allocator<Fxx, M> + Allocator<Fxx, N>,
{
    fn num_inputs(&self) -> usize;
    fn num_outputs(&self) -> usize;

    fn predict(&self, x: &VectorN<Fxx, M>) -> VectorN<Fxx, N>;
    fn update(&mut self, x: &VectorN<Fxx, M>, y: &VectorN<Fxx, N>) -> ();
}

#[derive(Clone, Copy, Debug)]
pub struct LinearModel<M, N> where 
    M: DimName + DimAdd<U1>,
    N: DimName,
    DefaultAllocator: Allocator<Fxx, M> + Allocator<Fxx, N> +
                      Allocator<Fxx, N, M>,
    Owned<Fxx, N>: Copy,
    Owned<Fxx, N, M>: Copy,
{
    ws: MatrixMN<Fxx, N, M>,
    bs: VectorN<Fxx, N>,
}

impl<M, N> LinearModel<M, N> where
    M: DimName + DimAdd<U1>,
    N: DimName,
    DefaultAllocator: Allocator<Fxx, M> + Allocator<Fxx, N> +
                      Allocator<Fxx, N, M>,
    Owned<Fxx, N>: Copy,
    Owned<Fxx, N, M>: Copy,
{
    pub fn new() -> Self {
        let m = LinearModel {
            ws: MatrixMN::<Fxx, N, M>::new_random(),
            bs: VectorN::<Fxx, N>::new_random(),
        };
        m
    }

    pub fn merge(&mut self, a: Fxx, other: &LinearModel<M,N>) -> () {
        self.ws = (1.0-a)*self.ws + a*other.ws;
        self.bs = (1.0-a)*self.bs + a*other.bs;
    }

    pub fn num_inputs(&self) -> usize {
        M::dim()
    }

    pub fn num_outputs(&self) -> usize {
        N::dim()
    }

    pub fn predict(&self, x: &VectorN<Fxx, M>) -> VectorN<Fxx, N> {
        self.ws * x + self.bs
    }

    pub fn update(&mut self, x: &VectorN<Fxx, M>, y: &VectorN<Fxx, N>) -> ()
        where DefaultAllocator: Allocator<Fxx, U1, M>
    {
        let update_size: Fxx = 0.25 / (M::dim() as Fxx); // XXX constant?
        let yh = self.predict(x);
        let err = yh - y;
        let deltas = 2.0 * update_size * err;
        self.bs = self.bs - deltas;
        self.ws = self.ws - deltas * x.transpose();
    }

    pub fn update_bulk<D: DimName>(&mut self,
                                   x: &MatrixMN<Fxx, M, D>,
                                   y: &MatrixMN<Fxx, N, D>) -> () where 
        DefaultAllocator: Reallocator<Fxx, M, D, DimSum<M, U1>, D> +
            Reallocator<Fxx, DimSum<M, U1>, D, D, DimSum<M, U1>> +
            Allocator<Fxx, N, D> +
            Allocator<Fxx, N, DimSum<M, U1>> +
            Allocator<Fxx, DimSum<M, U1>, DimSum<M, U1>> +
            Allocator<Fxx, D, M> + Allocator<Fxx, M, D> + Allocator<Fxx, M, M>,
        Owned<Fxx, M, D>: Copy,
        Owned<Fxx, D, DimSum<M, U1>>: Copy,
        Owned<Fxx, DimSum<M, U1>, D>: Copy,
    {
        let x1 = x.insert_row(M::dim(), 1.0); // M+1 x D
        let x1t = x1.transpose();             // D x M+1
        let xxt = x1 * x1t;                   // M+1 x M+1

        if let Some(xxt1) = xxt.try_inverse() {
            let w1 = y * x1t * xxt1; // N x M+1, the last column represents b
            self.bs = w1.column(M::dim()).into_owned();
            self.ws = MatrixMN::<Fxx, N, M>::from_column_slice(
                w1.columns(0, M::dim()).into_owned().as_slice());
        } else {
            panic!("cannot update_bulk, no inverse")
        }
    }
}

#[cfg(test)]
#[path = "./model_test.rs"]
mod model_test;
