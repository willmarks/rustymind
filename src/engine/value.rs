use std::fmt;

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
}

impl Node {
    pub fn new(data: f32, name: String, nodes: &mut Vec<Node>) -> usize {
        let out = Node {
            data,
            name,
            ..Default::default()
        };
        return out.push_on(nodes);
    }

    fn push_on(mut self, nodes: &mut Vec<Node>) -> usize {
        let idx = nodes.len();
        self._idx = idx;
        nodes.push(self);
        return idx;
    }
}

pub fn tanh(idx: usize, nodes: &mut Vec<Node>) -> usize {
    let x = nodes[idx].data;
    let t = ((2.0 * x).exp() - 1.0) / ((2.0 * x).exp() + 1.0);

    let out = Node {
        data: t,
        name: format!("tanh({})", nodes[idx].name),
        grad: nodes[idx].grad,
        _idx: 0,
        _prev: (Some(idx), None),
        _op: Op::Tanh,
    };
    out.push_on(nodes)
}

pub fn add(idx: usize, other: usize, nodes: &mut Vec<Node>) -> usize {
    let s = &nodes[idx];
    let other = &nodes[other];
    let out = Node {
        data: s.data + other.data,
        name: format!("({}+{})", s.name, other.name),
        grad: 0.0,
        _idx: 0,
        _prev: (Some(idx), Some(other._idx)),
        _op: Op::Add,
    };
    out.push_on(nodes)
}

pub fn sub(idx: usize, other: usize, nodes: &mut Vec<Node>) -> usize {
    let neg = Node {
        data: -1.0,
        name: String::from("neg"),
        ..Default::default()
    };
    let neg_idx = neg.push_on(nodes);

    add(idx, mul(other, neg_idx, nodes), nodes)
}

pub fn mul(idx: usize, other: usize, nodes: &mut Vec<Node>) -> usize {
    let s = &nodes[idx];
    let other = &nodes[other];
    let out = Node {
        data: s.data * other.data,
        name: format!("({}*{})", s.name, other.name),
        grad: 0.0,
        _idx: 0,
        _prev: (Some(idx), Some(other._idx)),
        _op: Op::Mul,
    };
    out.push_on(nodes)
}

pub fn exp(idx: usize, n: f32, nodes: &mut Vec<Node>) -> usize {
    let s = &nodes[idx];
    let out = Node {
        data: s.data.powf(n),
        name: format!("({}^{})", s.name, n),
        grad: 0.0,
        _idx: 0,
        _prev: (Some(idx), None),
        _op: Op::Exp(n),
    };
    out.push_on(nodes)
}

pub fn back(idx: usize, nodes: &mut Vec<Node>) {
    match nodes[idx]._op {
        Op::Add => back_add(idx, nodes),
        Op::Mul => back_mul(idx, nodes),
        Op::Exp(n) => back_exp(idx, n, nodes),
        Op::Tanh => back_tanh(idx, nodes),
        Op::NoOp => (), //println!("NoOp"),
    }
}

fn back_add(o: usize, nodes: &mut Vec<Node>) {
    // println!("back_add");
    match nodes[o]._prev {
        (Some(a), Some(b)) => {
            nodes[a].grad += 1.0 * nodes[o].grad;
            nodes[b].grad += 1.0 * nodes[o].grad;
            back(b, nodes);
            back(a, nodes);
        }
        _ => panic!("+ attempting to back propagate incorrectly"),
    }
}

fn back_mul(o: usize, nodes: &mut Vec<Node>) {
    // println!("back_mul");
    match nodes[o]._prev {
        (Some(a), Some(b)) => {
            nodes[a].grad += nodes[b].data * nodes[o].grad;
            nodes[b].grad += nodes[a].data * nodes[o].grad;
            back(b, nodes);
            back(a, nodes);
        }
        _ => panic!("* attempting to back propagate incorrectly"),
    }
}

fn back_exp(o: usize, n: f32, nodes: &mut Vec<Node>) {
    // println!("back_mul");
    match nodes[o]._prev {
        (Some(s), None) => {
            nodes[s].grad += n * nodes[s].data.powf(n - 1.0) * nodes[o].grad;
            back(s, nodes);
        }
        _ => panic!("^ attempting to back propagate incorrectly"),
    }
}

fn back_tanh(o: usize, nodes: &mut Vec<Node>) {
    // println!("back_tanh");
    match nodes[o]._prev {
        (Some(s), None) => {
            let x = nodes[s].data;
            let t = ((2.0 * x).exp() - 1.0) / ((2.0 * x).exp() + 1.0);
            nodes[s].grad += (1.0 - t.powf(2.0)) * nodes[o].grad;
            back(s, nodes);
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
        };
    }
}

impl fmt::Debug for Node {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "[{}: ({},{})]", self.name, self.data, self.grad)
    }
}
