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

    let prev_state: Vec<f32> = vec![0.921767, 0.44346145, -0.21449693, 0.26847574, -0.30725732, 1.2618302, 0.5746452, 0.9046577, 0.17796758, -0.5812532, 2.1809006, 0.5675714, 0.549642, 0.41012135, 1.024522, 0.5665747, -0.7112087, -0.9252703, -0.035549056, -0.92275405, -0.02088232, 0.89397293, -0.3086933, 1.3669239, 0.021202061, -0.69741535, 0.10025542, -0.19920866, -0.79703, -0.62239075, 0.9060309, 0.0056635803, -0.40449753, 0.7041124, 0.69651675, -0.6350977, 0.06863236, -1.4438491, 1.2940397, -0.7627458, -0.36631885];

    mlp.set_state(prev_state, &mut nodes);

    let start = Instant::now();
    mlp.train(&xs, &ys, 1000, 0.01, &mut nodes);
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
