use std::time::Instant;

use crate::{mind::mlp::MLP, engine::state::State};

mod engine;
mod mind;

fn main() {
    let mut state = State {nodes: vec![]};

    let xs1 = vec![(2.0f32, "x11"), (3.0f32, "x12"), (-1.0f32, "x13")];
    let xs2 = vec![(3.0f32, "x21"), (-1.0f32, "x22"), (0.5f32, "x23")];
    let xs3 = vec![(0.5f32, "x31"), (1.0f32, "x32"), (1.0f32, "x33")];
    let xs4 = vec![(1.0f32, "x41"), (1.0f32, "x42"), (-1.0f32, "x43")];

    let xs = vec![xs1, xs2, xs3, xs4];
    let ys = vec![1.0f32, -1.0f32, -1.0f32, 1.0f32];

    let mut mlp = MLP::new(3, vec![4, 4, 1], &mut state);

    let prev_state: Vec<f32> = vec![-0.49166176, -0.96705574, 1.467671, -0.46258911, -1.4749879, -0.28841418, -0.8427304, 0.4273841, 0.48191345, -0.2928523, -0.013751007, 0.5748077, -1.2170222, 0.110074684, 0.6232349, -0.68140614, -0.5193674, 0.8242914, -0.1759994, -0.5249664, 1.2176541];
    mlp.set_state(prev_state, &mut state);

    let start = Instant::now();
    mlp.train(&xs, &ys, 10000, 0.01, &mut state);
    let duration = start.elapsed();

    println!("duration: {:?}", duration);
    println!("state:\n{:?}", mlp.get_state(&state));
}

#[macro_export]
macro_rules! str {
    () => {
        String::new()
    };
    ($x:expr $(,)?) => {
        ToString::to_string(&$x)
    };
}
