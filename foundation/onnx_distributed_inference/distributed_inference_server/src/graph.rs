use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use uuid::Uuid;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Node {
    pub id: Uuid,
    pub next_nodes: Vec<Uuid>,
    pub data: Value,
    #[serde(skip)]
    users: Vec<Uuid>,
}

impl Node {
    pub fn new(next_nodes: Vec<Uuid>, data: Value) -> Self {
        Self {
            id: Uuid::new_v4(),
            next_nodes,
            data,
            users: Vec::new(),
        }
    }
}

#[derive(Debug)]
pub struct Graph {
    nodes: HashMap<Uuid, Node>,
    pub start_node_id: Uuid,
}

impl Graph {
    pub fn new(nodes: Vec<Node>) -> Self {
        Self {
            start_node_id: nodes[0].id.clone(),
            nodes: nodes.into_iter().map(|node| (node.id, node)).collect(),
        }
    }

    pub fn add_user(&mut self, user_id: Uuid) -> Option<Uuid> {
        // Assign user to start node by default
        if self.nodes.is_empty() {
            return None;
        }

        self.nodes
            .iter_mut()
            .filter(|(_, node)| node.users.is_empty())
            .next()
            .map(|(id, node)| {
                node.users.push(user_id);
                *id
            })
    }

    pub fn remove_user(&mut self, user_id: &Uuid) {
        for node in self.nodes.values_mut() {
            node.users.retain(|id| id != user_id);
        }
    }

    pub fn get_node(&self, node_id: &Uuid) -> Option<&Node> {
        self.nodes.get(node_id)
    }

    pub fn get_next_nodes(&self, node_id: &Uuid) -> Vec<&Node> {
        self.nodes
            .get(node_id)
            .unwrap()
            .next_nodes
            .iter()
            .map(|id| self.nodes.get(id).unwrap())
            .collect()
    }

    pub fn get_worker(&self, node_id: &Uuid) -> Option<Uuid> {
        if let Some(node) = self.nodes.get(node_id) {
            return node.users.first().cloned();
        }
        None
    }
}
