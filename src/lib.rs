#![crate_name = "lair"]

mod img;
pub use img::conv2d::Conv2d;
pub use img::{
    range_matrix, range_vector, read_luma, read_lumad, write_luma_matrix, write_luma_vector,
};

mod layered_model;
pub use layered_model::LayeredModel;

mod linear_model;
pub use linear_model::LinearModel;

mod logit;
pub use logit::Logit;

mod model;
pub use model::Fxx;
pub use model::Model;

mod relu;
pub use relu::Relu;

mod trainer;
pub use trainer::BatchTrainer;
pub use trainer::GradientTrainer;
pub use trainer::MomentumTrainer;
pub use trainer::SGDTrainer;
pub use trainer::UpdateParams;
