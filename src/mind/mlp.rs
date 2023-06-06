use crate::engine::{
    node::{add, back, exp, sub, Node},
    state::State,
};

use super::layer::Layer;

#[derive(Debug)]
pub struct MLP {
    layers: Vec<Layer>,
    parameters: Vec<usize>,
}

impl MLP {
    pub fn new(n_in: u32, layer_sizes: Vec<u32>, state: &mut State) -> MLP {
        let mut layers = vec![];

        let layer = Layer::new(n_in, layer_sizes[0], format!("ins"), state);
            layers.push(layer);

        for i in 1..layer_sizes.len() - 1 {
            let layer = Layer::new(layer_sizes[i], layer_sizes[i + 1], format!("{}", i), state);
            layers.push(layer);
        }

        let mut parameters = vec![];
        for layer in layers.iter().rev() {
            for neuron in layer.neurons.iter() {
                parameters.push(neuron.bias);
                for weight in neuron.weights.iter() {
                    parameters.push(*weight);
                }
            }
        }

        MLP { layers, parameters }
    }

    pub fn apply(&mut self, xs: &Vec<(f32, &str)>, state: &mut State) -> (Vec<usize>, Vec<usize>) {
        let mut inputs = vec![];
        for (data, name) in xs {
            inputs.push(Node::new(*data, String::from(*name), state));
        }

        let mut outputs: Vec<usize> = inputs;
        let mut checkpoints: Vec<usize> = vec![];
        for layer in self.layers.iter() {
            outputs = layer.apply(&outputs, state);
            checkpoints.append(&mut outputs.to_vec());
        }

        (outputs, checkpoints.into_iter().rev().collect())
    }

    pub fn get_state(&self, state: &State) -> Vec<f32> {
        self.parameters.iter().map(|n| state[*n].data).collect()
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
        }
    }

    fn zero_grad(&mut self, state: &mut State) {
        for param in self.parameters.iter() {
            state[*param].grad = 0.0;
        }
    }

    fn truncate_nodes(&self, state: &mut State) {
        state.nodes.truncate(self.parameters.len())
    }

    pub fn train(
        &mut self,
        inputs: &Vec<Vec<(f32, &str)>>,
        expected: &Vec<f32>,
        iterations: i32,
        step: f32,
        state: &mut State,
    ) {
        for iteration in 1..=iterations {
            let mut outs: Vec<usize> = vec![];
            let mut checkpoints: Vec<Vec<usize>> = vec![];

            for i in 0..inputs.len() {
                let (output, checks) = self.apply(&inputs[i], state);
                outs.push(output[0]);
                checkpoints.push(checks);
            }

            let loss = loss(expected, &outs, state);

            if iteration < iterations {
                self.zero_grad(state);
                state[loss].grad = 1.0f32;
                back(loss, state);

                for i in 0..checkpoints[0].len() {
                    for apply_checks in checkpoints.iter() {
                        // println!("Applying: {}", state[apply_checks[i]].name);
                        back(apply_checks[i], state)
                    }
                }

                self.learn(state, step);
                self.truncate_nodes(state);
            } else {
                println!("{}: {}", iteration, state[loss].data);
                let res: Vec<f32> = outs.iter().map(|o| state[*o].data).collect();
                println!("outs: {:?}", res);
            }
        }
    }
}

/// Calculates the loss by summing the squared the differences between actual and expected
pub fn loss(expect: &Vec<f32>, actual: &Vec<usize>, state: &mut State) -> usize {
    let mut exs = vec![];
    for e in expect {
        exs.push(Node::new(*e, format!("{{e:{}}}", e), state));
    }

    let mut out: Option<usize> = None;
    for (e_val, a_val) in exs.into_iter().zip(actual.into_iter()) {
        let sub_res = sub(*a_val, e_val, state);
        let exp_res = exp(sub_res, 2.0f32, state);
        out = match out {
            Some(v) => Some(add(v, exp_res, state)),
            None => Some(exp_res),
        }
    }
    return out.unwrap();
}
