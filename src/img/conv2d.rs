extern crate nalgebra as na;

use na::{DimAdd, DimDiff, DimMul, DimName, DimProd, DimSub, DimSum};
use na::{VectorN};
use na::{U1};
use na::allocator::Allocator;
use na::DefaultAllocator;
use na::storage::Owned;

use crate::model::{Fxx, Model};

///
/// Generics wrapper around an operation to summarize a PoolRow x PoolCol x PoolInputs
/// matrix to a PoolOutputs summary.
/// 
pub struct Pooler2d<'a, PoolRow, PoolCol, PoolInputs, PoolOutputs>
where
    PoolRow: DimName + DimMul<PoolCol>,
    PoolCol: DimName,
    PoolInputs: DimName + DimMul<DimProd<PoolRow, PoolCol>>,
    PoolOutputs: DimName,
{
    model: &'a mut dyn Model<
        DimProd<PoolInputs, DimProd<PoolRow, PoolCol>>, 
        PoolOutputs>,
}

///
/// 2D convolution operator.
/// 
pub struct Conv2d<'a, Pr, Pc, Pi, Po, Ir, Ic>
where
    Pr: DimName + DimMul<Pc>,
    Pc: DimName,
    Pi: DimName + DimMul<DimProd<Pr, Pc>>,
    Po: DimName + DimMul<DimProd<DimSum<DimDiff<Ir, Pr>, U1>, DimSum<DimDiff<Ic, Pc>, U1>>>,
    Ir: DimName + DimSub<Pr>,
    Ic: DimName + DimSub<Pc>,
    // Co: DimName + DimMul<DimProd<DimSum<DimDiff<Ir, Pr>, U1>, DimSum<DimDiff<Ic, Pc>, U1>>>,
    DimDiff<Ir, Pr>: DimAdd<U1>,
    DimDiff<Ic, Pc>: DimAdd<U1>,
    DimSum<DimDiff<Ir, Pr>, U1>: DimMul<DimSum<DimDiff<Ic, Pc>, U1>>,
{
    pooler: Pooler2d<'a, Pr, Pc, Pi, Po>,
    model: Option<&'a dyn Model< // make PhantomData
        DimProd<Po, DimProd<DimSum<DimDiff<Ir, Pr>, U1>, DimSum<DimDiff<Ic, Pc>, U1>>>,
        DimProd<DimSum<DimDiff<Ir, Pr>, U1>, DimSum<DimDiff<Ic, Pc>, U1>>>>,
}

impl <'a, Pr, Pc, Pi, Po, Ir, Ic> Conv2d<'a, Pr, Pc, Pi, Po, Ir, Ic>
where
    Pr: DimName + DimMul<Pc>,
    Pc: DimName,
    Pi: DimName + DimMul<DimProd<Pr, Pc>>,
    Po: DimName + DimMul<DimProd<DimSum<DimDiff<Ir, Pr>, U1>, DimSum<DimDiff<Ic, Pc>, U1>>>,
    Ir: DimName + DimSub<Pr>,
    Ic: DimName + DimSub<Pc>,
    // Co: DimName + DimMul<DimProd<DimSum<DimDiff<Ir, Pr>, U1>, DimSum<DimDiff<Ic, Pc>, U1>>>,
    DimDiff<Ir, Pr>: DimAdd<U1>,
    DimDiff<Ic, Pc>: DimAdd<U1>,
    DimSum<DimDiff<Ir, Pr>, U1>: DimMul<DimSum<DimDiff<Ic, Pc>, U1>>,
{
    pub fn new(pooler: &'a mut dyn Model<DimProd<Pi, DimProd<Pr, Pc>>, Po>) -> Self {
        Conv2d {
            pooler: Pooler2d::<Pr, Pc, Pi, Po> {
                model: pooler,
            },
            model: None,
        }
    }
}

#[cfg(test)]
#[path = "./conv2d_test.rs"]
mod conv2d_test;
