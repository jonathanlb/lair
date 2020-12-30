#![crate_name = "lair"]

mod linear_model;
pub use linear_model::LinearModel;

mod model;
pub use model::Fxx;
pub use model::Model;

mod relu;
pub use relu::Relu;
