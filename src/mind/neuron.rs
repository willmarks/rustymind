use std::fmt;

use rand::Rng;

use crate::engine::{node::{add, mul, tanh, Node}, state::State};

pub struct Neuron {
    weights: Vec<usize>,
    bias: usize,
}

impl Neuron {
    pub fn new(n_in: u32, state: &mut State) -> Neuron {
        let mut rng = rand::thread_rng();

        let mut weights = vec![];
        for n in 1..=n_in {
            let w_idx = Node::new(rng.gen_range(-1.0..1.0), format!("w{}", n), true, state);
            weights.push(w_idx);
        }

        let bias = Node::new(rng.gen_range(-1.0..1.0), String::from("b"), true, state);

        Neuron { weights, bias }
    }

    pub fn parameters(&self) -> Vec<usize> {
        let mut params = self.weights.clone();
        params.push(self.bias);
        return params;
    }
}

pub fn apply(neuron: &Neuron, xs: &Vec<usize>, state: &mut State) -> usize {
    let mut act: Option<usize> = None;
    for i in 0..neuron.weights.len() {
        match act {
            Some(v) => {
                act = Some(add(v, mul(neuron.weights[i], xs[i], state), state));
            }
            None => {
                act = Some(mul(neuron.weights[i], xs[i], state));
            }
        }
    }

    tanh(add(act.unwrap(), neuron.bias, state), state)
}

impl fmt::Debug for Neuron {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}:{:?}", self.weights, self.bias)
    }
}
