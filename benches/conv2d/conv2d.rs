use criterion::Criterion;
use rand::distributions::{Distribution, Normal, Uniform};

use std::fs;
use std::io;
use std::path::{Path, PathBuf};

extern crate lair;
extern crate nalgebra as na;

use lair::{
    range_matrix, range_vector, read_luma, read_lumad, write_luma_matrix, write_luma_vector,
};
use lair::{Conv2d, Fxx, LayeredModel, LinearModel, Logit, Model, Relu, SGDTrainer, UpdateParams};
use na::allocator::Allocator;
use na::storage::Owned;
use na::{DefaultAllocator, VectorN};
use na::{DimDiff, DimName, DimProd, DimSum, U1, U32, U4};
use typenum::{U300, U400};

struct TrainParams<'a> {
    num_targets: (f64, f64), // mean, stdev
    sz_targets: (f64, f64),
    test_backs: &'a Path,
    test_targets: &'a Path,
    train_backs: &'a Path,
    train_targets: &'a Path,
}

type Dim2d = (usize, usize);
type Width = U400;
type Height = U300;
type InputD = DimProd<Width, Height>;
type PoolS = U32;
type Pool0 = U4;
// number of outputs for the first layer: 4*(1+400-32)*(1+300-32)
type Output0Rows = DimSum<U1, DimDiff<Height, PoolS>>;
type Output0Cols = DimSum<U1, DimDiff<Width, PoolS>>;
type OutputD0 = DimProd<Pool0, DimProd<Output0Cols, Output0Rows>>;

type Output1Rows = DimSum<U1, DimDiff<Output0Rows, PoolS>>;
type Output1Cols = DimSum<U1, DimDiff<Output0Cols, PoolS>>;
type OutputD1 = DimProd<Output1Rows, Output1Cols>;

const LEARNING_PARAMS: UpdateParams = UpdateParams {
    step_size: 1e-4,
    l2_reg: 0.0,
};

fn choose_img(dir: &Path) -> io::Result<PathBuf> {
    fn is_img<'r>(entry: &'r Result<fs::DirEntry, io::Error>) -> bool {
        match entry {
            Ok(f) => f.path().ends_with(".png"),
            _ => false,
        }
    }

    if !dir.is_dir() {
        Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("not a directory: {}", dir.to_string_lossy()),
        ))
    } else {
        let count = dir.read_dir()?.filter(is_img).count();
        if count <= 0 {
            Err(io::Error::new(
                io::ErrorKind::NotFound,
                format!("no valid image found: {}", dir.to_string_lossy()),
            ))
        } else {
            let dist = Uniform::from(0..count);
            let choice = dist.sample(&mut rand::thread_rng());
            dir.read_dir()?
                .filter(is_img)
                .nth(choice)
                .ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::NotFound,
                        format!("no valid image found: {}", dir.to_string_lossy()),
                    )
                })
                .map(|entry| entry.unwrap().path())
        }
    }
}

fn compose_example<M, N>(
    target_img: &Vec<(PathBuf, Dim2d)>,
    back_img: (&Path, Dim2d),
) -> (VectorN<Fxx, M>, VectorN<Fxx, N>)
where
    M: DimName,
    N: DimName,
    DefaultAllocator: Allocator<Fxx, M> + Allocator<Fxx, N>,
    Owned<Fxx, M>: Copy,
    Owned<Fxx, N>: Copy,
{
    let mut back_mat = read_luma::<_, Height, Width>(&mut io::BufReader::new(
        fs::File::open(back_img.0.as_os_str()).unwrap(),
    ))
    .unwrap();
    let mut positions = VectorN::<Fxx, N>::zeros();

    for i in target_img {
        let x_dist = Uniform::from(0..Width::dim() - i.1 .1);
        let x = x_dist.sample(&mut rand::thread_rng());
        let y_dist = Uniform::from(0..Width::dim() - i.1 .0);
        let y = y_dist.sample(&mut rand::thread_rng());
        let target_mat = read_lumad(
            &mut io::BufReader::new(fs::File::open(i.0.as_os_str()).unwrap()),
            i.1,
        )
        .unwrap();
        // Copy the target into the background
        for row in 0..y {
            let row_off = row + i.1 .0;
            for col in 0..x {
                let col_off = col + i.1 .1;
                back_mat[(row_off, col_off)] = target_mat[(row, col)];
            }
        }

        // calculate ouput positions.
        let y_off = (y as f64
            + 0.5 * (i.1 .0 as f64) * (Output1Rows::dim() as f64) / (Height::dim() as f64))
            .round()
            .max(Output1Rows::dim() as f64 - 1.0)
            .min(0.0) as usize;
        let x_off = (x as f64
            + 0.5 * (i.1 .1 as f64) * (Output1Cols::dim() as f64) / (Width::dim() as f64))
            .round()
            .max(Output1Cols::dim() as f64 - 1.0)
            .min(0.0) as usize;
        positions[(y as usize + y_off) * Output1Cols::dim() + (x as usize + x_off)] = 1.0;
    }

    let back_rows =
        VectorN::<Fxx, M>::from_fn(|i, _| back_mat[(i / Width::dim(), i % Width::dim())]);
    (back_rows, positions)
}

fn create_example<M, N>(params: &TrainParams) -> (VectorN<Fxx, M>, VectorN<Fxx, N>)
where
    M: DimName,
    N: DimName,
    DefaultAllocator: Allocator<Fxx, M> + Allocator<Fxx, N>,
    Owned<Fxx, M>: Copy,
    Owned<Fxx, N>: Copy,
{
    let norm_num = Normal::new(params.num_targets.0, params.num_targets.1);
    let norm_sz = Normal::new(params.sz_targets.0, params.sz_targets.1);
    let num_targets = norm_num.sample(&mut rand::thread_rng()).round().max(0.0) as usize;
    let targets = [0..num_targets]
        .iter()
        .map(|_| {
            let sz = norm_sz.sample(&mut rand::thread_rng()).round().max(1.0) as usize;
            (choose_img(params.test_targets).unwrap(), (sz, sz))
        })
        .collect();
    let back_img = choose_img(params.test_backs).unwrap(); // XXX test or train? and above
    let back_dim = (Height::dim(), Width::dim());
    compose_example(&targets, (&back_img, back_dim))
}

fn find_target(params: &TrainParams) {
    let mut train0 = SGDTrainer::new(&LEARNING_PARAMS);
    let mut pooler0 = LinearModel::<DimProd<PoolS, PoolS>, Pool0>::new_random(&mut train0);
    let mut cnn0 =
        Conv2d::<PoolS, PoolS, U1, Pool0, Height, Width, InputD, OutputD0>::new(&mut pooler0);
    let mut layer0 = Relu::new(&mut cnn0);

    let mut train1 = SGDTrainer::new(&LEARNING_PARAMS);
    let mut pooler1 =
        LinearModel::<DimProd<Pool0, DimProd<PoolS, PoolS>>, U1>::new_random(&mut train1);
    let mut cnn1 =
        Conv2d::<PoolS, PoolS, Pool0, U1, Output0Rows, Output0Cols, OutputD0, OutputD1>::new(
            &mut pooler1,
        );
    let mut layer1 = Logit::new(&mut cnn1);

    let mut model = LayeredModel::new(&mut layer0, &mut layer1);

    let examples = [0..64]
        .iter()
        .map(|_| create_example::<InputD, OutputD1>(params));
    examples.for_each(|ex| {
        model.update(&ex.0, &ex.1);
    });

    // Evaluate, log error XXX

    // Output pooler states
    for i in 0..Pool0::dim() {
        let path = format!("ws0{}.png", i);
        let slice = pooler0.get_ws().row(i).transpose();
        write_luma_vector::<DimProd<PoolS, PoolS>>(
            &range_vector(&slice),
            PoolS::dim(),
            PoolS::dim(),
            &path,
        )
        .unwrap();
    }
    write_luma_matrix(&range_matrix(pooler1.get_ws()), &"ws1.png").unwrap();
}

pub fn find_targets(c: &mut Criterion) {
    let params = TrainParams {
        num_targets: (1.0, 1.0),
        sz_targets: (32.0, 5.0),
        test_backs: &Path::new("./benches/conv2d/train_backs"),
        test_targets: &Path::new("./benches/conv2d/train_targets"),
        train_backs: &Path::new("./benches/conv2d/train_backs"),
        train_targets: &Path::new("./benches/conv2d/train_targets"),
    };

    c.bench_function("find_simple", |b| b.iter(|| find_target(&params)));
}
