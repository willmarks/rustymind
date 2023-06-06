use crate::engine::state::State;

use super::neuron::Neuron;

#[derive(Debug)]
pub struct Layer {
    pub neurons: Vec<Neuron>,
}

impl Layer {
    pub fn new(n_in: u32, n_out: u32, name: String, state: &mut State) -> Layer {
        let mut neurons = vec![];

        for n in 0..n_out {
            neurons.push(Neuron::new(n_in, format!("l{}n{}", name, n), state))
        }

        return Layer { neurons };
    }

    pub fn apply(&self, xs: &Vec<usize>, state: &mut State) -> Vec<usize> {
        let mut outs = vec![];
        for neuron in self.neurons.iter() {
            outs.push(neuron.apply(xs, state));
        }
        return outs;
    }
}
