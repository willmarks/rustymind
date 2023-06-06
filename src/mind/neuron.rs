use std::fmt;

use rand::Rng;

use crate::engine::{
    node::{add, checkpoint, mul, tanh, Node},
    state::State,
};

pub struct Neuron {
    pub name: String,
    pub weights: Vec<usize>,
    pub bias: usize,
}

impl Neuron {
    pub fn new(n_in: u32, name: String, state: &mut State) -> Neuron {
        let mut rng = rand::thread_rng();

        let mut weights = vec![];
        for n in 1..=n_in {
            let w_idx = Node::new(rng.gen_range(-1.0..1.0), format!("{}w{}", name, n), state);
            weights.push(w_idx);
        }

        let bias = Node::new(rng.gen_range(-1.0..1.0), format!("{}b", name), state);

        Neuron {
            name,
            weights,
            bias,
        }
    }

    /// Takes inputs and constructs the node equivalent of the eval fn to allow for back propagation.
    /// Wraps all of its nodes in a checkpoint to prevent back propagation before upstream nodes
    /// have been back propagated.
    pub fn apply(&self, xs: &Vec<usize>, state: &mut State) -> usize {
        let mut act: Option<usize> = None;
        for (i, w) in self.weights.iter().enumerate() {
            let mul_res = mul(*w, xs[i], state);
            match act {
                Some(v) => {
                    act = Some(add(v, mul_res, state));
                }
                None => {
                    act = Some(mul_res);
                }
            }
        }

        checkpoint(
            tanh(add(act.unwrap(), self.bias, state), state),
            &self.name,
            state,
        )
    }

    /// Evaluates the inputs with the weights stored in the state.
    pub fn eval(&self, xs: &Vec<f32>, state: &State) -> f32 {
        let mut act: f32 = 0.0;
        for (i, w) in self.weights.iter().enumerate() {
            act += state[*w].data * xs[i];
        }
        let x = act + state[self.bias].data;
        return ((2.0 * x).exp() - 1.0) / ((2.0 * x).exp() + 1.0);
    }
}

impl fmt::Debug for Neuron {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}[{:?}:{:?}]", self.name, self.weights, self.bias)
    }
}
