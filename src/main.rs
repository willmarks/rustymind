use crate::{engine::value::back, mind::mlp::{MLP, loss}};

// use mind::neuron::Neuron;

mod engine;
mod mind;

fn main() {
    let mut nodes = vec![];

    let xs1 = vec![(2.0f32, "x11"),(3.0f32, "x12"),(-1.0f32,"x13")];
    let xs2 = vec![(3.0f32, "x21"),(-1.0f32, "x22"),(0.5f32, "x23")];
    let xs3 = vec![(0.5f32, "x31"),(1.0f32, "x32"),(1.0f32, "x33")];
    let xs4 = vec![(1.0f32, "x41"),(1.0f32, "x42"),(-1.0f32, "x43")];

    let xs = vec![xs1, xs2, xs3, xs4];
    let ys = vec![1.0f32, -1.0f32, -1.0f32, 1.0f32];

    // TODO: dump old values when finished back prop with them?
    let mlp = MLP::new(3, vec![4, 4, 1], &mut nodes);

    mlp.train(&xs, &ys, 1000, 0.01, &mut nodes);
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
