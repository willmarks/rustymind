use crate::engine::value::{add, back, exp, sub, Node};

use super::layer::{apply as layer_apply, Layer};

#[derive(Debug)]
pub struct MLP {
    layers: Vec<Layer>,
    pub params: Vec<usize>,
}

impl MLP {
    pub fn new(n_in: u32, n_outs: Vec<u32>, nodes: &mut Vec<Node>) -> MLP {
        let mut layers = vec![];
        let mut params = vec![];

        let input_layer = Layer::new(n_in, n_outs[0], nodes);
        params.append(&mut input_layer.parameters());
        layers.push(input_layer);
  
        for i in 1..n_outs.len() {
            let layer = Layer::new(n_outs[i - 1], n_outs[i], nodes);
            params.append(&mut layer.parameters());
            layers.push(layer);
        }

         MLP { layers, params }
    }

    pub fn apply(&self, xs: &Vec<(f32, &str)>, nodes: &mut Vec<Node>) -> Vec<usize> {
        let mut xsi = vec![];
        for (data, name) in xs {
            xsi.push(Node::new(*data, String::from(*name), nodes));
        }

        let mut out: Option<Vec<usize>> = None;
        for layer_idx in 0..self.layers.len() {
            out = match out {
                Some(v) => Some(layer_apply(&self.layers[layer_idx], &v, nodes)),
                None => Some(layer_apply(&self.layers[layer_idx], &xsi, nodes)),
            }
        }

        out.unwrap()
    }

    pub fn get_state(&self, nodes: &Vec<Node>) -> Vec<f32> {
        self.parameters()
            .into_iter()
            .map(|n| nodes[n].data)
            .collect()
    }

    pub fn set_state(&self, weights: Vec<f32>, nodes: &mut Vec<Node>) {
        if weights.len() != nodes.len() {
            print!("Error loading state");
        }

        for (w, n) in weights.into_iter().zip(self.parameters().into_iter()) {
            nodes[n].data = w
        }
    }

    fn parameters(&self) -> Vec<usize> {
        let mut params = vec![];
        for i in 0..self.layers.len() {
            params.append(&mut self.layers[i].parameters());
        }
        return params;
    }

    fn learn(&self, nodes: &mut Vec<Node>, step: f32) {
        for param in self.parameters() {
            // let old = nodes[param].data;
            // let new = old + nodes[param].grad * -step;
            // println!("old: {}, grad: {}, new: {}", old, nodes[param].grad, new);
            nodes[param].data += nodes[param].grad * -step
        }
    }

    fn zero_grad(&self, nodes: &mut Vec<Node>) {
        for param in self.parameters() {
            nodes[param].grad = 0.0;
        }
    }

    fn truncate_nodes(&self, nodes: &mut Vec<Node>) {
        nodes.truncate(self.params.len())
    }

    pub fn train(
        &self,
        inputs: &Vec<Vec<(f32, &str)>>,
        expected: &Vec<f32>,
        iterations: i32,
        step: f32,
        nodes: &mut Vec<Node>,
    ) {
        for iteration in 1..=iterations {
            let mut outs: Vec<usize> = vec![];
            for i in 0..inputs.len() {
                outs.push(self.apply(&inputs[i], nodes)[0]);
            }

            let loss = loss(expected, &outs, nodes);

            if iteration < iterations {
                self.zero_grad(nodes);
                nodes[loss].grad = 1.0f32;
                back(loss, nodes);
                self.truncate_nodes(nodes)
            } else {
                println!("{}: {}", iteration, nodes[loss].data);
                let res: Vec<f32> = outs.iter().map(|o| nodes[*o].data).collect();
                println!("outs: {:?}", res);
            }

            self.learn(nodes, step);
        }
    }
}

pub fn loss(expect: &Vec<f32>, actual: &Vec<usize>, nodes: &mut Vec<Node>) -> usize {
    let mut exs = vec![];
    for e in expect {
        exs.push(Node::new(*e, String::from("e"), nodes));
    }

    let mut out: Option<usize> = None;
    for (e_val, a_val) in exs.into_iter().zip(actual.into_iter()) {
        let sub_res = sub(*a_val, e_val, nodes);
        let exp_res = exp(sub_res, 2.0f32, nodes);
        out = match out {
            Some(v) => {
                let add_res = add(v, exp_res, nodes);
                Some(add_res)
            }
            None => Some(exp_res),
        }
    }
    return out.unwrap();
}
