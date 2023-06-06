use std::fmt;

use super::state::State;

pub enum Op {
    Add,
    Mul,
    Tanh,
    Exp(f32),
    Pass,
    End,
}

pub struct Node {
    pub data: f32,
    pub name: String,
    pub grad: f32,
    pub _idx: usize,
    pub _prev: (Option<usize>, Option<usize>),
    pub _op: Op,
    pub _parameter: bool,
}

impl Node {
    pub fn new(data: f32, name: String, is_parameter: bool, state: &mut State) -> usize {
        let out = Node {
            data,
            name,
            _parameter: is_parameter,
            ..Default::default()
        };
        return out.push_on(state);
    }

    fn push_on(mut self, state: &mut State) -> usize {
        let idx = state.nodes.len();
        self._idx = idx;
        state.nodes.push(self);
        return idx;
    }
}

pub fn checkpoint(node: usize, name: &str, state: &mut State) -> usize {
    let out = Node {
        data: state[node].data,
        name: format!("Checkpoint({})", name),
        _prev: (Some(node), None),
        _op: Op::Pass,
        ..Default::default()
    };
    out.push_on(state)
}

pub fn tanh(node: usize, state: &mut State) -> usize {
    let x = state[node].data;
    let t = ((2.0 * x).exp() - 1.0) / ((2.0 * x).exp() + 1.0);
    let out = Node {
        data: t,
        name: format!("tanh({})", state[node].name),
        _prev: (Some(node), None),
        _op: Op::Tanh,
        ..Default::default()
    };
    out.push_on(state)
}

pub fn add(node: usize, other: usize, state: &mut State) -> usize {
    let s = &state[node];
    let other = &state[other];
    let out = Node {
        data: s.data + other.data,
        name: format!("({}+{})", s.name, other.name),
        _prev: (Some(node), Some(other._idx)),
        _op: Op::Add,
        ..Default::default()
    };
    out.push_on(state)
}

pub fn sub(node: usize, other: usize, state: &mut State) -> usize {
    let neg = Node {
        data: -1.0,
        name: String::from("neg"),
        ..Default::default()
    };
    let neg_idx = neg.push_on(state);

    add(node, mul(other, neg_idx, state), state)
}

pub fn mul(node: usize, other: usize, state: &mut State) -> usize {
    let s = &state[node];
    let other = &state[other];
    let out = Node {
        data: s.data * other.data,
        name: format!("({}*{})", s.name, other.name),
        _prev: (Some(node), Some(other._idx)),
        _op: Op::Mul,
        ..Default::default()
    };
    out.push_on(state)
}

pub fn exp(node: usize, n: f32, state: &mut State) -> usize {
    let s = &state[node];
    let out = Node {
        data: s.data.powf(n),
        name: format!("({}^{})", s.name, n),
        _prev: (Some(node), None),
        _op: Op::Exp(n),
        ..Default::default()
    };
    out.push_on(state)
}

pub fn back(node: usize, state: &mut State) {
    // println!("back: {}", state[node].name);
    match state[node]._op {
        Op::Add => back_add(node, state),
        Op::Mul => back_mul(node, state),
        Op::Exp(n) => back_exp(node, n, state),
        Op::Tanh => back_tanh(node, state),
        Op::Pass => back_pass(node, state),
        Op::End => ()
    }
}

/**
 * Computes the backward pass until it reaches a checkpoint
 */
fn _back(node: usize, state: &mut State) {
    match state[node]._op {
        Op::Pass => (), // println!("- Stopping: {}", state[node].name),
        _ => back(node, state)
    }
}

fn back_pass(o: usize, state: &mut State) {
    match state[o]._prev {
        (Some(a), None) => {
            state[a].grad += state[o].grad;
            _back(a, state);
        }
        _ => panic!("-> attempting to back propagate incorrectly"),
    }
}

fn back_add(o: usize, state: &mut State) {
    // println!("back_add");
    match state[o]._prev {
        (Some(a), Some(b)) => {
            state[a].grad += 1.0 * state[o].grad;
            state[b].grad += 1.0 * state[o].grad;
            _back(b, state);
            _back(a, state);
        }
        _ => panic!("+ attempting to back propagate incorrectly"),
    }
}

fn back_mul(o: usize, state: &mut State) {
    // println!("back_mul");
    match state[o]._prev {
        (Some(a), Some(b)) => {
            state[a].grad += state[b].data * state[o].grad;
            state[b].grad += state[a].data * state[o].grad;
            _back(b, state);
            _back(a, state);
        }
        _ => panic!("* attempting to back propagate incorrectly"),
    }
}

fn back_exp(o: usize, n: f32, state: &mut State) {
    // println!("back_mul");
    match state[o]._prev {
        (Some(s), None) => {
            state[s].grad += n * state[s].data.powf(n - 1.0) * state[o].grad;
            _back(s, state);
        }
        _ => panic!("^ attempting to back propagate incorrectly"),
    }
}

fn back_tanh(o: usize, state: &mut State) {
    // println!("back_tanh");
    match state[o]._prev {
        (Some(s), None) => {
            let x = state[s].data;
            let t = ((2.0 * x).exp() - 1.0) / ((2.0 * x).exp() + 1.0);
            state[s].grad += (1.0 - t.powf(2.0)) * state[o].grad;
            _back(s, state);
        }
        _ => panic!("tanh attempting to back propagate incorrectly"),
    }
}

impl Default for Node {
    fn default() -> Self {
        return Node {
            data: 0.0,
            grad: 0.0,
            name: String::new(),
            _idx: 0,
            _prev: (None, None),
            _op: Op::End,
            _parameter: false,
        };
    }
}

impl fmt::Debug for Node {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "[{}: ({},{})]", self.name, self.data, self.grad)
    }
}
