use std::collections::HashMap;
use uuid::Uuid;

use crate::protos::ModelConfig;

#[derive(Clone, Debug)]
pub struct Node {
    pub id: Uuid,
    pub next_node: Option<Uuid>,
    pub data: ModelConfig,
}

impl Node {
    pub fn new(next_node: Option<Uuid>, data: ModelConfig) -> Self {
        Self {
            id: Uuid::new_v4(),
            next_node,
            data,
        }
    }
}

#[derive(Debug)]
pub struct StructuralGraph {
    pub start_node_id: Uuid,
    nodes: HashMap<Uuid, Node>,
}

impl StructuralGraph {
    pub fn new(nodes: Vec<Node>) -> Self {
        Self {
            start_node_id: nodes[0].id,
            nodes: nodes.into_iter().map(|node| (node.id, node)).collect(),
        }
    }

    pub fn list_nodes(&self) -> impl Iterator<Item = &Uuid> {
        self.nodes.keys()
    }

    pub fn get_node(&self, node_id: &Uuid) -> Option<&Node> {
        self.nodes.get(node_id)
    }

    pub fn get_next_node(&self, node_id: &Uuid) -> Option<Uuid> {
        self.nodes.get(node_id).unwrap().next_node
    }
}
