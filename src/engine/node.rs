use std::fmt;

use super::state::State;

pub enum Op {
    Add,
    Mul,
    Tanh,
    Exp(f32),
    NoOp,
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

pub fn tanh(idx: usize, state: &mut State) -> usize {
    let x = state[idx].data;
    let t = ((2.0 * x).exp() - 1.0) / ((2.0 * x).exp() + 1.0);
    let out = Node {
        data: t,
        name: format!("tanh({})", state[idx].name),
        grad: state[idx].grad,
        _idx: 0,
        _prev: (Some(idx), None),
        _op: Op::Tanh,
        ..Default::default()
    };
    out.push_on(state)
}

pub fn add(idx: usize, other: usize, state: &mut State) -> usize {
    let s = &state[idx];
    let other = &state[other];
    let out = Node {
        data: s.data + other.data,
        name: format!("({}+{})", s.name, other.name),
        _prev: (Some(idx), Some(other._idx)),
        _op: Op::Add,
        ..Default::default()
    };
    out.push_on(state)
}

pub fn sub(idx: usize, other: usize, state: &mut State) -> usize {
    let neg = Node {
        data: -1.0,
        name: String::from("neg"),
        ..Default::default()
    };
    let neg_idx = neg.push_on(state);

    add(idx, mul(other, neg_idx, state), state)
}

pub fn mul(idx: usize, other: usize, state: &mut State) -> usize {
    let s = &state[idx];
    let other = &state[other];
    let out = Node {
        data: s.data * other.data,
        name: format!("({}*{})", s.name, other.name),
        _prev: (Some(idx), Some(other._idx)),
        _op: Op::Mul,
        ..Default::default()
    };
    out.push_on(state)
}

pub fn exp(idx: usize, n: f32, state: &mut State) -> usize {
    let s = &state[idx];
    let out = Node {
        data: s.data.powf(n),
        name: format!("({}^{})", s.name, n),
        _prev: (Some(idx), None),
        _op: Op::Exp(n),
        ..Default::default()
    };
    out.push_on(state)
}

pub fn back(idx: usize, state: &mut State) {
    if !state[idx]._parameter {
        match state[idx]._op {
            Op::Add => back_add(idx, state),
            Op::Mul => back_mul(idx, state),
            Op::Exp(n) => back_exp(idx, n, state),
            Op::Tanh => back_tanh(idx, state),
            Op::NoOp => (), //println!("NoOp"),
        }
    }    
}

fn back_add(o: usize, state: &mut State) {
    // println!("back_add");
    match state[o]._prev {
        (Some(a), Some(b)) => {
            state[a].grad += 1.0 * state[o].grad;
            state[b].grad += 1.0 * state[o].grad;
            back(b, state);
            back(a, state);
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
            back(b, state);
            back(a, state);
        }
        _ => panic!("* attempting to back propagate incorrectly"),
    }
}

fn back_exp(o: usize, n: f32, state: &mut State) {
    // println!("back_mul");
    match state[o]._prev {
        (Some(s), None) => {
            state[s].grad += n * state[s].data.powf(n - 1.0) * state[o].grad;
            back(s, state);
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
            back(s, state);
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
            _op: Op::NoOp,
            _parameter: false,
        };
    }
}

impl fmt::Debug for Node {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "[{}: ({},{})]", self.name, self.data, self.grad)
    }
}
