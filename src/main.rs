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

    let mut mlp = MLP::new(vec![3, 3, 3, 1], &mut state);

    let prev_state: Vec<f32> = vec![-0.5841904, -0.08590077, -0.5293272, 0.5594426, 0.9479385, -0.43416417, -0.34560978, 0.061313838, -0.7083538, 0.784774, 0.3480366, 0.12489496, 0.25960645, 0.8229814, 0.80480343, -0.43075958, -0.0586538, 0.15411803, 0.20576242, -0.6518703, 0.26415867, -0.5469353, 0.32332993, -0.37897536, 0.027082179, -0.9807686, 0.86140454, -0.031825673];
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
