use std::fmt;

use rand::Rng;

use crate::engine::value::{add, mul, tanh, Node};

pub struct Neuron {
    weights: Vec<usize>,
    bias: usize,
}

impl Neuron {
    pub fn new(n_in: u32, nodes: &mut Vec<Node>) -> Neuron {
        let mut rng = rand::thread_rng();

        let mut weights = vec![];
        for n in 1..=n_in {
            let w_idx = Node::new(rng.gen_range(-1.0..1.0), format!("w{}", n), nodes);
            weights.push(w_idx);
        }

        let bias = Node::new(rng.gen_range(-1.0..1.0), String::from("b"), nodes);

        Neuron { weights, bias }
    }

    pub fn parameters(&self) -> Vec<usize> {
        let mut params = self.weights.clone();
        params.push(self.bias);
        return params;
    }
}

pub fn apply(neuron: &Neuron, xs: &Vec<usize>, nodes: &mut Vec<Node>) -> usize {
    let mut act: Option<usize> = None;
    for i in 0..neuron.weights.len() {
        match act {
            Some(v) => {
                act = Some(add(v, mul(neuron.weights[i], xs[i], nodes), nodes));
            }
            None => {
                act = Some(mul(neuron.weights[i], xs[i], nodes));
            }
        }
    }

    tanh(add(act.unwrap(), neuron.bias, nodes), nodes)
}

impl fmt::Debug for Neuron {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}:{:?}", self.weights, self.bias)
    }
}
