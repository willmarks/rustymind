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

    /// Applies the inputs to each neuron and returns the vec of their output nodes.
    pub fn apply(&self, xs: &Vec<usize>, state: &mut State) -> Vec<usize> {
        let mut outs = vec![];
        for neuron in self.neurons.iter() {
            outs.push(neuron.apply(xs, state));
        }
        return outs;
    }

    /// Evaluates the inputs for each neuron and returns their results.
    pub fn eval(&self, xs: &Vec<f32>, state: &State) -> Vec<f32> {
        let mut outs = vec![];
        for neuron in self.neurons.iter() {
            outs.push(neuron.eval(xs, state));
        }
        return outs;
    } 
}
