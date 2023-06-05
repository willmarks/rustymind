use std::ops::{IndexMut, Index};

use super::node::Node;

pub struct State {
    pub nodes: Vec<Node>
}

impl Index<usize> for State {
    type Output = Node;
    fn index(&self, node: usize) -> &Node {
        &self.nodes[node]
    }
}

impl IndexMut<usize> for State {
    fn index_mut(&mut self, node: usize) -> &mut Node {
        &mut self.nodes[node]
    }
}