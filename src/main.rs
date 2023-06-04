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

    let prev_state: Vec<f32> = vec![0.9201422, 0.43847683, -0.22002608, 0.2647854, -0.30252573, 1.2557144, 0.5798789, 0.9038458, 0.1735063, -0.5840546, 2.1630352, 0.56489116, 0.60133725, 0.4822256, 0.977071, 0.57565033, -0.7111242, -0.9253895, -0.035365038, -0.9226384, -0.020789618, 0.89402574, -0.3090107, 1.3571289, 0.019316526, -0.69858813, 0.10347583, -0.2020933, -0.7775513, -0.61639047, 0.91124016, 0.0042040977, -0.40549466, 0.69353163, 0.6931707, -0.638, 0.07232908, -1.4201578, 1.2766724, -0.74471617, -0.36923763];

    mlp.set_state(prev_state, &mut nodes);

    let start = Instant::now();
    mlp.train(&xs, &ys, 1000, 0.01, &mut nodes);
    let duration = start.elapsed();

    println!("duration: {:?}", duration);
    println!("state: {:?}", mlp.get_state(&nodes));
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
