# LAIR
LAIR (Learning Artificial Intelligence with Rust) is a set of implementation
exercises I use to demonstrate (to myself) gaps in my understanding in
machine learning -- at least linear regression and deep learning so far.

## Debugging
LAIR uses the [log](https://docs.rs/log/0.4.14/log/) and 
[env_logger](https://docs.rs/env_logger/0.8.2/env_logger/) crates for logging.

The code is also sprinkled with 
[debug assertions](https://doc.rust-lang.org/std/macro.debug_assert.html) 
for use in tracking down numerical instability.

Compile/run/etc with
```
RUST_LOG=debug RUST_BACKTRACE=1 RUSTFLAGS="-g -C debug-assertions" cargo ...
```

## Benchmarks
There are some simple benchmarks I use for run-time (convergence) performance
produced by

```
CRITERIUM_DEBUG=1 cargo bench
```

## TODOs
- Implement and benchmark classification.
- Implement sigmoid function for backpropagation.
- Test and benchmark with use of ReLU and sigmoid functions.
- Implement meta-parameterized search (separate training from model).
 - Batch updates.
- Add (parameterized) noise to benchmarks.
- Improve numerical instability.
 - Dynamic step size/learning rate.
- Benchmark convergence.
