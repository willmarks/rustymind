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

    let xs: Vec<Vec<(f32, &str)>> = vec![xs1, xs2, xs3, xs4];
    let ys = vec![-0.6f32, -0.15f32, 0.05f32, -0.1f32];

    let mut mlp = MLP::new(3, vec![10, 10, 10, 1], &mut state);

    let prev_state: Vec<f32> = vec![-0.72789097, -0.030183518, -0.22515628, -0.6869764, 0.02587848, 0.53538907, 0.7876665, -0.86576986, 0.06715777, 0.9564611, -0.4570492, 0.1920475, -0.72010314, 0.90961194, -0.86335653, -0.8666198, -0.32010967, 0.9678414, -0.15705886, 0.39817825, -0.167633, 0.10914272, -0.85943663, -0.16158882, 0.23317152, 0.1278755, -0.05448236, 0.4452646, -0.48774794, 0.31464654, 0.69742084, -0.8821304, -0.46198323, -0.70982975, -0.4753613, -0.31708407, -0.6915348, -0.7526486, 0.45153245, -0.8923659, -0.9547703, 0.5709665, 0.9339072, -0.73212874, -0.4006993, -0.22631335, -0.34534842, 0.51624846, -0.058783878, 0.56722325, 0.42468715, 0.8407204, -0.4274505, 0.06416239, 0.7811647, -0.48696363, 0.38352117, 0.9246391, -0.32393247, -0.73766685, -0.7710576, 0.28664088, 0.59584755, 0.4925274, -0.4202868, 0.11982408, -0.9899269, 0.41901708, 0.72359157, -0.2548694, -0.48810953, 0.6681286, 0.2994992, -0.8123988, -0.7392746, 0.5524583, -0.9311394, -0.6294188, -0.74963754, 0.23705372, 0.6040393, 0.6515585, -0.18112701, 0.5209893, -0.9454915, 0.29813978, 0.19573928, -0.9264498, -0.08953654, -0.83850586, 0.52732086, -0.87213075, -0.7498846, -0.49573708, 0.37987548, -0.36165264, 0.060239684, 0.19610249, -0.7782573, 0.30424142, 0.33365455, -0.073892996, -0.05822007, 0.9660342, 0.0724696, 0.54045266, -0.22826917, 0.9590429, -0.34618413, -0.74225277, -0.33270693, 0.90071553, -0.27253124, 0.050914492, -0.16925098, -0.54872125, 0.6457287, 0.849666, 0.017286086, -0.8963935, -0.026266303, -0.51389295, -0.79841, 0.404808, 0.6258727, -0.80595946, -0.08866604, 0.6952122, 0.23265104, 0.55189204, 0.5649279, -0.6503835, 0.53804964, -0.46908453, 0.43720463, -0.27842543, -0.91416574, 0.6105827, 0.8086751, 0.26815164, -0.7115057, 0.58617204, 0.19978382, 0.4011151, 0.032175723, 0.386837, 0.0966064, -0.13716939, 0.5415264, -0.72121966, 0.24120703, 0.42021093, -0.6387923, 0.0045268033, -0.056451008, 0.26096708, 0.7641838, -0.5485483, -0.32352623, -0.55144393, -0.33391985, 0.07520249, 0.5901765, -0.73107314, 0.9750617, -0.9913681, 0.25716028, -0.90287304, 0.691937, -0.5940771, -0.75495696, -0.202116, -0.17630646, -0.94916624, 0.2526217, 0.21334477, -0.49742708, -0.49761313, -0.78693986, -0.7126779, 0.36580145, -0.081099436, 0.9661421, 0.6637046, -0.43367907, -0.4909354, -0.6319839, 0.104508474, -0.8700408, -0.3428732, -0.44586194, 0.52908665, 0.32385725, -0.8960624, -0.7734737, -0.21410841, -0.75669175, -0.4692186, 0.62910306, -0.375464, -0.5636153, -0.07020013, 0.8470509, -0.63786453, -0.27961317, -0.92672575, 0.44942912, -0.7606592, -0.011724281, -0.92135435, -0.18774219, 0.51992, -0.51024, -0.77309084, 0.22196451, 0.8459622, -0.6089677, -0.7914561, -0.3936701, 0.09556162, -0.91898817, 0.96348566, -0.94736236, -0.049858417, 0.3795896, -1.0041072, 0.96246, -0.6737346, -0.01966276, -0.42481658, 0.5936723, -0.7263087, -0.33800638, 0.5019825, -0.09751073, 0.45537043, 0.6182208, 0.3719057, 0.27115223, 0.958511, 0.6922592, -0.27001143, 0.9228965, 0.73948526, -0.0024032113, 0.641764, 0.82578915, 0.14156394, -0.21809159, 0.12954618, 0.6920467, 0.45653898, -0.13785443, -0.50051725, -0.5063716, 0.3248101, -0.1390033, 0.105190806, 0.65717924, 0.57773423, -0.88188285, 0.9273748, -0.4678831, -0.9804597, -0.5733138, 0.80755734, 0.33133745, -0.46594334, -0.06279051, -1.0079373, -0.55726326];
    mlp.set_state(prev_state, &mut state);

    let training_start = Instant::now();
    mlp.train(&xs, &ys, 1, 0.01, &mut state);
    println!("Finished training in {:?}", training_start.elapsed());

    let unknown = vec![3.0f32, 1.0f32, 1.0f32];
    let eval_start = Instant::now();
    let res = mlp.eval(unknown, &state);
    println!("Eval: [{}] - ({:?})", res[0], eval_start.elapsed());

    let weights_biases = mlp.get_state(&state);
    println!("state: {}\n{:?}", weights_biases.len(), weights_biases);
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
