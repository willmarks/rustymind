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

    let mut mlp = MLP::new(vec![3, 4, 4, 1], &mut state);

    let prev_state: Vec<f32> = vec![0.05395535, 0.47196275, -1.5572982, 0.66498417, 0.81236345, -0.30865374, 1.004092, 0.5767586, -0.6584805, -0.7781684, 0.2920248, -0.14075619, -0.42781702, -1.3901552, 0.21025907, 0.22390099, -0.17025706, -0.53297365, 0.92656064, -0.93411744, 0.15357432, 0.37622285, -0.59730756, 0.30281022, -1.0781792, 0.50017864, -0.4738097, 0.42222342, -1.0235, -0.788351, 0.2025136, 0.62342083, 0.087770164, 0.5562215, -0.76908267, 0.24191196, -1.2027358, 1.1013685, -0.20283732, -0.7992402, 0.5928649];
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
