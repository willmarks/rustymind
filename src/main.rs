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

    let prev_state: Vec<f32> = vec![0.5229019, -0.6020257, 1.6912806, 0.94682604, 1.0144699, 0.8527321, 0.19592991, 0.00786721, 0.0069333436, 0.9086944, 0.15276036, 0.43469483, -1.2679355, 0.22291589, -0.11801494, -0.6196133, 0.49736026, -0.37130305, -1.0163401, -0.4023145, 0.088479206, 1.140499, -0.27056444, -0.15039882, -0.44633183, 0.39085302, -0.24146551, 0.18032613, -0.8643691, 0.15420938, 0.030936485, -0.39116833, 1.1375767, -0.118813865, -0.58822215, -0.6798096, 0.1723934, 0.3825498, 0.5554118, 0.872463, -0.1062012];
    mlp.set_state(prev_state, &mut state);

    let start = Instant::now();
    mlp.train(&xs, &ys, 10000, 0.01, &mut state);
    let duration = start.elapsed();

    println!("duration: {:?}", duration);
    println!("state:\n{:?}", mlp.get_state(&state).len());
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
