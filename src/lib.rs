#![crate_name = "lair"]

pub mod img;
pub use img::conv2d::Conv2d;

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

use std::sync::Once;

static INIT: Once = Once::new();

/// Setup function that is only run once, even if called multiple times.
/// https://stackoverflow.com/questions/30177845/how-to-initialize-the-logger-for-integration-tests
pub fn setup_logging() {
    INIT.call_once(|| {
        env_logger::init();
    });
}
