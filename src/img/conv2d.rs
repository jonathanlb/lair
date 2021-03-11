extern crate nalgebra as na;

use na::allocator::Allocator;
use na::constraint::{DimEq, ShapeConstraint};
use na::storage::{Owned, Storage, StorageMut};
use na::DefaultAllocator;
use na::{Matrix, VectorN};
use na::U1;
use na::{DimAdd, DimDiff, DimMul, DimName, DimProd, DimSub, DimSum};
use std::marker::PhantomData;

use log::debug;

use crate::model::{Fxx, Model};

///
/// 2D convolution operator.
///
pub struct Conv2d<'a, Pr, Pc, Pi, Po, Ir, Ic, M, N>
where
    Pr: DimName + DimMul<Pc>,
    Pc: DimName,
    Pi: DimName + DimMul<DimProd<Pr, Pc>> + DimMul<DimProd<Ir, Ic>>,
    Po: DimName + DimMul<DimProd<DimSum<DimDiff<Ir, Pr>, U1>, DimSum<DimDiff<Ic, Pc>, U1>>>,
    Ir: DimName + DimSub<Pr> + DimMul<Ic>,
    Ic: DimName + DimSub<Pc>,
    M: DimName,
    N: DimName,
    ShapeConstraint: DimEq<M, DimProd<Pi, DimProd<Ir, Ic>>>
        + DimEq<N, DimProd<Po, DimProd<DimSum<DimDiff<Ir, Pr>, U1>, DimSum<DimDiff<Ic, Pc>, U1>>>>,
    DimDiff<Ir, Pr>: DimAdd<U1>,
    DimDiff<Ic, Pc>: DimAdd<U1>,
    DimSum<DimDiff<Ir, Pr>, U1>: DimMul<DimSum<DimDiff<Ic, Pc>, U1>>,
    DimProd<Pi, DimProd<Pr, Pc>>: DimName,
    DefaultAllocator: Allocator<Fxx, M>
        + Allocator<Fxx, N>
        + Allocator<Fxx, Pi>
        + Allocator<Fxx, Po>
        + Allocator<Fxx, DimProd<Pi, DimProd<Ir, Ic>>>
        + Allocator<Fxx, DimProd<Pi, DimProd<Pr, Pc>>>,
    Owned<Fxx, Pi>: Copy,
    Owned<Fxx, Po>: Copy,
    Owned<Fxx, DimProd<Pi, DimProd<Pr, Pc>>>: Copy,
{
    pooler: &'a mut dyn Model<DimProd<Pi, DimProd<Pr, Pc>>, Po>,
    _model_impl: PhantomData<&'a dyn Model<M, N>>,
    _model_spec: PhantomData<
        &'a dyn Model<
            DimProd<Pi, DimProd<Ir, Ic>>,
            DimProd<Po, DimProd<DimSum<DimDiff<Ir, Pr>, U1>, DimSum<DimDiff<Ic, Pc>, U1>>>,
        >,
    >,
}

impl<'a, Pr, Pc, Pi, Po, Ir, Ic, M, N> Conv2d<'a, Pr, Pc, Pi, Po, Ir, Ic, M, N>
where
    Pr: DimName + DimMul<Pc>,
    Pc: DimName,
    Pi: DimName + DimMul<DimProd<Pr, Pc>> + DimMul<DimProd<Ir, Ic>>,
    Po: DimName + DimMul<DimProd<DimSum<DimDiff<Ir, Pr>, U1>, DimSum<DimDiff<Ic, Pc>, U1>>>,
    Ir: DimName + DimSub<Pr> + DimMul<Ic>,
    Ic: DimName + DimSub<Pc>,
    M: DimName,
    N: DimName,
    ShapeConstraint: DimEq<M, DimProd<Pi, DimProd<Ir, Ic>>>
        + DimEq<N, DimProd<Po, DimProd<DimSum<DimDiff<Ir, Pr>, U1>, DimSum<DimDiff<Ic, Pc>, U1>>>>,
    DimDiff<Ir, Pr>: DimAdd<U1>,
    DimDiff<Ic, Pc>: DimAdd<U1>,
    DimSum<DimDiff<Ir, Pr>, U1>: DimMul<DimSum<DimDiff<Ic, Pc>, U1>>,
    DimProd<Pi, DimProd<Pr, Pc>>: DimName,
    DefaultAllocator: Allocator<Fxx, M>
        + Allocator<Fxx, N>
        + Allocator<Fxx, Pi>
        + Allocator<Fxx, Po>
        + Allocator<Fxx, DimProd<Pi, DimProd<Ir, Ic>>>
        + Allocator<Fxx, DimProd<Pi, DimProd<Pr, Pc>>>,
    Owned<Fxx, Pi>: Copy,
    Owned<Fxx, Po>: Copy,
    Owned<Fxx, DimProd<Pi, DimProd<Pr, Pc>>>: Copy,
{
    pub fn new(pooler: &'a mut dyn Model<DimProd<Pi, DimProd<Pr, Pc>>, Po>) -> Self {
        Conv2d {
            pooler: pooler,
            _model_impl: PhantomData,
            _model_spec: PhantomData,
        }
    }

    ///
    /// Extract/transform the input into the patch used by the pooler at rxc.
    ///
    fn get_input_patch<S>(
        &self,
        input: &Matrix<Fxx, M, U1, S>,
        r: usize,
        c: usize,
    ) -> VectorN<Fxx, DimProd<Pi, DimProd<Pr, Pc>>>
    where S: Storage<Fxx, M, U1>
    {
        let pc = Pc::dim();
        let pr = Pr::dim();
        let pi = Pi::dim();
        let ic = Ic::dim();
        let input_rows_offset = r * Ic::dim() * pi;
        let input_cols_offset = c * pi;

        let mut patch = VectorN::<Fxx, DimProd<Pi, DimProd<Pr, Pc>>>::zeros();
        for i in 0..pr {
            for j in 0..pc {
                for k in 0..pi {
                    patch[pi * (i * pc + j) + k] =
                        input[input_rows_offset + input_cols_offset + pi * (i * ic + j) + k];
                }
            }
        }
        patch
    }

    ///
    /// Get the portion of the output contributed to by the pooler at input rxc.
    ///
    fn get_output_error_patch<S>(
        &self,
        err: &Matrix<Fxx, N, U1, S>,
        r: usize,
        c: usize,
    ) -> VectorN<Fxx, Po> 
    where S: Storage<Fxx, N, U1>
    {
        let oc = Ic::dim() - Pc::dim() + 1;
        let po = Po::dim();
        let offset = po * (r * oc + c);
        VectorN::<Fxx, Po>::from_fn(|i, _| err[offset + i])
    }

    ///
    /// Copy the output from the rxc pool into the destination.
    ///
    fn patch_output<S0, S1>(
        &self,
        pooled: &Matrix<Fxx, Po, U1, S0>,
        dest: &mut Matrix<Fxx, N, U1, S1>,
        r: usize,
        c: usize,
    ) -> () 
    where
        S0: Storage<Fxx, Po, U1>,
        S1: StorageMut<Fxx, N, U1>,
    {
        let po = Po::dim();
        let oc = Ic::dim() - Pc::dim() + 1;
        let offset = po * (r * oc + c);
        for i in 0..po {
            dest[offset + i] = pooled[i];
        }
    }

    ///
    /// Take the error produced by backpropogation from the pooler at rxc and copy it into the
    /// destination for pooling backpropogation error.
    ///
    fn patch_error(
        &self,
        error: &VectorN<Fxx, DimProd<Pi, DimProd<Pr, Pc>>>,
        pooled_error: &mut VectorN<Fxx, M>,
        r: usize,
        c: usize,
    ) -> () {
        let pc = Pc::dim();
        let pr = Pr::dim();
        let pi = Pi::dim();
        let ic = Ic::dim();
        let input_rows_offset = r * Ic::dim() * pi;
        let input_cols_offset = c * pi;

        for i in 0..pr {
            for j in 0..pc {
                for k in 0..pi {
                    let src_off = pi * (i * pc + j) + k;
                    let dest_off = input_rows_offset + input_cols_offset + pi * (i * ic + j) + k;
                    pooled_error[dest_off] += error[src_off];
                }
            }
        }
    }
}

///
/// Implement 2d convolution as a model.
/// Currently there might be a bunch of unneccessary copying of data.  The slices from input and
/// output data are immutable, but are copied with into_owned() because conevert isn't implemented
/// for into().
///
impl<'a, Pr, Pc, Pi, Po, Ir, Ic, M, N> Model<M, N> for Conv2d<'a, Pr, Pc, Pi, Po, Ir, Ic, M, N>
where
    Pr: DimName + DimMul<Pc>,
    Pc: DimName,
    Pi: DimName + DimMul<DimProd<Pr, Pc>> + DimMul<DimProd<Ir, Ic>>,
    Po: DimName + DimMul<DimProd<DimSum<DimDiff<Ir, Pr>, U1>, DimSum<DimDiff<Ic, Pc>, U1>>>,
    Ir: DimName + DimSub<Pr> + DimMul<Ic>,
    Ic: DimName + DimSub<Pc>,
    M: DimName,
    N: DimName,
    ShapeConstraint: DimEq<M, DimProd<Pi, DimProd<Ir, Ic>>>
        + DimEq<N, DimProd<Po, DimProd<DimSum<DimDiff<Ir, Pr>, U1>, DimSum<DimDiff<Ic, Pc>, U1>>>>,
    DimDiff<Ir, Pr>: DimAdd<U1>,
    DimDiff<Ic, Pc>: DimAdd<U1>,
    DimSum<DimDiff<Ir, Pr>, U1>: DimMul<DimSum<DimDiff<Ic, Pc>, U1>>,
    DimProd<Pi, DimProd<Pr, Pc>>: DimName,
    DefaultAllocator: Allocator<Fxx, M>
        + Allocator<Fxx, N>
        + Allocator<Fxx, Pi>
        + Allocator<Fxx, Po>
        + Allocator<Fxx, DimProd<Pi, DimProd<Ir, Ic>>>
        + Allocator<Fxx, DimProd<Pi, DimProd<Pr, Pc>>>,
    Owned<Fxx, Pi>: Copy,
    Owned<Fxx, Po>: Copy,
    Owned<Fxx, DimProd<Pi, DimProd<Pr, Pc>>>: Copy,
{
    #[inline]
    fn num_inputs(&self) -> usize {
        M::dim()
    }

    #[inline]
    fn num_outputs(&self) -> usize {
        N::dim()
    }

    fn backpropagate(&mut self, x: &VectorN<Fxx, M>, de_dy: &VectorN<Fxx, N>) -> VectorN<Fxx, M>
    where
        DefaultAllocator: Allocator<Fxx, N> + Allocator<Fxx, M>,
    {
        let mut de_dx = VectorN::<Fxx, M>::zeros();
        for r in 0..1 {
            for c in 0..1 {
                let err_patch = self.get_output_error_patch(de_dy, r, c);
                let sub_x = self.get_input_patch(x, r, c);
                let sub_result = self.pooler.backpropagate(&sub_x, &err_patch);
                self.patch_error(&sub_result, &mut de_dx, r, c);
            }
        }
        de_dx
    }

    fn predict(&self, x: &VectorN<Fxx, M>) -> VectorN<Fxx, N>
    where
        DefaultAllocator: Allocator<Fxx, M> + Allocator<Fxx, N>,
    {
        let mut y = VectorN::<Fxx, N>::zeros();
        debug!(
            "M={}, N={}, Pr={}, Pc={}, Pi={}, Po={}, Ir={}, Ic={}",
            M::dim(),
            N::dim(),
            Pr::dim(),
            Pc::dim(),
            Pi::dim(),
            Po::dim(),
            Ir::dim(),
            Ic::dim()
        );
        for r in 0..(1 + Ir::dim() - Pr::dim()) {
            for c in 0..(1 + Ic::dim() - Pc::dim()) {
                let sub_image = self.get_input_patch(x, r, c);
                let sub_result = self.pooler.predict(&sub_image);
                self.patch_output(&sub_result, &mut y, r, c);
            }
        }
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
#[path = "./conv2d_test.rs"]
mod conv2d_test;
