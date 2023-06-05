use crate::engine::{node::{add, back, exp, sub, Node}, state::State};

use super::layer::{apply as layer_apply, Layer};

#[derive(Debug)]
pub struct MLP {
    layers: Vec<Layer>,
    pub parameters: Vec<usize>,
}

impl MLP {
    pub fn new(n_in: u32, n_outs: Vec<u32>, state: &mut State) -> MLP {
        let mut layers = vec![];
        let mut parameters = vec![];

        let input_layer = Layer::new(n_in, n_outs[0], state);
        parameters.append(&mut input_layer.parameters());
        layers.push(input_layer);
  
        for i in 1..n_outs.len() {
            let layer = Layer::new(n_outs[i - 1], n_outs[i], state);
            parameters.append(&mut layer.parameters());
            layers.push(layer);
        }

         MLP { layers, parameters: parameters.into_iter().rev().collect() }
    }

    pub fn apply(&self, xs: &Vec<(f32, &str)>, state: &mut State) -> Vec<usize> {
        let mut xsi = vec![];
        for (data, name) in xs {
            xsi.push(Node::new(*data, String::from(*name), false, state));
        }

        let mut out: Option<Vec<usize>> = None;
        for layer_idx in 0..self.layers.len() {
            out = match out {
                Some(v) => Some(layer_apply(&self.layers[layer_idx], &v, state)),
                None => Some(layer_apply(&self.layers[layer_idx], &xsi, state)),
            }
        }

        out.unwrap()
    }

    pub fn get_state(&self, state: &State) -> Vec<f32> {
        self.parameters
            .iter()
            .map(|n| state[*n].data)
            .collect()
    }

    pub fn set_state(&self, weights: Vec<f32>, state: &mut State) {
        if weights.len() != state.nodes.len() {
            panic!("Error loading state");
        }

        for (w, n) in weights.into_iter().zip(self.parameters.iter()) {
            state[*n].data = w
        }
    }

    fn learn(&self, state: &mut State, step: f32) {
        for param in self.parameters.iter() {
            // let old = nodes[param].data;
            // let new = old + nodes[param].grad * -step;
            // println!("old: {}, grad: {}, new: {}", old, nodes[param].grad, new);
            state[*param].data += state[*param].grad * -step;
            back(*param, state);
        }
    }

    fn zero_grad(&self, state: &mut State) {
        for param in self.parameters.iter() {
            state[*param].grad = 0.0;
        }
    }

    fn truncate_nodes(&self, state: &mut State) {
        state.nodes.truncate(self.parameters.len())
    }

    pub fn train(
        &self,
        inputs: &Vec<Vec<(f32, &str)>>,
        expected: &Vec<f32>,
        iterations: i32,
        step: f32,
        state: &mut State,
    ) {
        for iteration in 1..=iterations {
            let mut outs: Vec<usize> = vec![];
            for i in 0..inputs.len() {
                outs.push(self.apply(&inputs[i], state)[0]);
            }

            let loss = loss(expected, &outs, state);

            if iteration < iterations {
                self.zero_grad(state);
                state[loss].grad = 1.0f32;
                back(loss, state);
                self.truncate_nodes(state)
            } else {
                println!("{}: {}", iteration, state[loss].data);
                let res: Vec<f32> = outs.iter().map(|o| state[*o].data).collect();
                println!("outs: {:?}", res);
            }

            self.learn(state, step);
        }
    }
}

pub fn loss(expect: &Vec<f32>, actual: &Vec<usize>, state: &mut State) -> usize {
    let mut exs = vec![];
    for e in expect {
        exs.push(Node::new(*e, String::from("e"), false, state));
    }

    let mut out: Option<usize> = None;
    for (e_val, a_val) in exs.into_iter().zip(actual.into_iter()) {
        let sub_res = sub(*a_val, e_val, state);
        let exp_res = exp(sub_res, 2.0f32, state);
        out = match out {
            Some(v) => {
                let add_res = add(v, exp_res, state);
                Some(add_res)
            }
            None => Some(exp_res),
        }
    }
    return out.unwrap();
}
