pub use self::conv2d::find_targets;

mod conv2d;

criterion_group!(conv2d, find_targets);
