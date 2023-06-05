use crate::engine::{state::State};

use super::neuron::{apply as neuron_apply, Neuron};

#[derive(Debug)]
pub struct Layer {
    neurons: Vec<Neuron>,
}

impl Layer {
    pub fn new(n_in: u32, n_out: u32, state: &mut State) -> Layer {
        let mut neurons = vec![];

        for _n in 0..n_out {
            neurons.push(Neuron::new(n_in, state))
        }

        return Layer { neurons };
    }

    pub fn parameters(&self) -> Vec<usize> {
        let mut params = vec![];
        for i in 0..self.neurons.len() {
            params.append(&mut self.neurons[i].parameters());
        }
        return params;
    }
}

pub fn apply(layer: &Layer, xs: &Vec<usize>, state: &mut State) -> Vec<usize> {
    let mut outs = vec![];
    for neuron_idx in 0..layer.neurons.len() {
        outs.push(neuron_apply(&layer.neurons[neuron_idx], xs, state));
    }
    return outs;
}
