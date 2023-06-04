use std::time::Instant;

use crate::mind::mlp::MLP;

mod engine;
mod mind;

fn main() {
    let mut nodes = vec![];

    let xs1 = vec![(2.0f32, "x11"), (3.0f32, "x12"), (-1.0f32, "x13")];
    let xs2 = vec![(3.0f32, "x21"), (-1.0f32, "x22"), (0.5f32, "x23")];
    let xs3 = vec![(0.5f32, "x31"), (1.0f32, "x32"), (1.0f32, "x33")];
    let xs4 = vec![(1.0f32, "x41"), (1.0f32, "x42"), (-1.0f32, "x43")];

    let xs = vec![xs1, xs2, xs3, xs4];
    // let xs = vec![xs1, xs2];
    let ys = vec![1.0f32, -1.0f32, -1.0f32, 1.0f32];
    // let ys = vec![1.0f32, -1.0f32];

    let mlp = MLP::new(3, vec![4, 4, 1], &mut nodes);

    let prev_state: Vec<f32> = vec![0.93688494, 0.4808476, -0.17761606, 0.28943804, -0.3393256, 1.2735462, 0.5560555, 0.90660566, 0.22234593, -0.53979206, 2.2547572, 0.5842103, 0.2587054, -0.11478697, 1.3170272, 0.5375873, -0.71137524, -0.92399734, -0.037146363, -0.92400575, -0.020879863, 0.9002244, -0.30137947, 1.4119401, 0.046947043, -0.68530524, 0.060063742, -0.20355408, -0.8770983, -0.6786296, 0.8562667, 0.028241366, -0.3851261, 0.75013536, 0.72979903, -0.6048995, 0.07810016, -1.5543696, 1.3816235, -0.8508475, -0.37622407];

    mlp.set_state(prev_state, &mut nodes);

    let start = Instant::now();
    mlp.train(&xs, &ys, 10000, 0.01, &mut nodes);
    let duration = start.elapsed();

    println!("duration: {:?}", duration);
    println!("state:\n{:?}", mlp.get_state(&nodes));
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
