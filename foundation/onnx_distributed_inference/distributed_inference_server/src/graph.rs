use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

use crate::ModelConfig;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Node {
    pub id: Uuid,
    pub next_nodes: Vec<Uuid>,
    pub data: ModelConfig,
    #[serde(skip)]
    users: Vec<(Uuid, Status)>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Status {
    Loading,
    Available,
}

impl Node {
    pub fn new(next_nodes: Vec<Uuid>, data: ModelConfig) -> Self {
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
            start_node_id: nodes[0].id,
            nodes: nodes.into_iter().map(|node| (node.id, node)).collect(),
        }
    }

    pub fn get_worker(&self, node_id: &Uuid) -> Option<Uuid> {
        self.nodes.get(node_id).and_then(|node| {
            node.users
                .iter()
                .find(|(_, status)| *status == Status::Available)
                .map(|(id, _)| id)
                .cloned()
        })
    }

    pub fn add_worker(&mut self, user_id: Uuid) -> Option<Uuid> {
        self.nodes
            .iter_mut()
            .find(|(_, node)| node.users.is_empty())
            .map(|(id, node)| {
                node.users.push((user_id, Status::Loading));
                *id
            })
    }

    pub fn enable_worker(&mut self, user_id: &Uuid, node_id: &Uuid) {
        let node = self.nodes.get_mut(node_id).unwrap();

        node.users
            .iter_mut()
            .filter(|(id, _)| id == user_id)
            .for_each(|(_, status)| *status = Status::Available);
    }

    pub fn remove_worker(&mut self, user_id: &Uuid) {
        self.nodes
            .values_mut()
            .for_each(|node| node.users.retain(|(id, _)| id != user_id))
    }

    pub fn get_node(&self, node_id: &Uuid) -> Option<&Node> {
        self.nodes.get(node_id)
    }

    pub fn get_next_nodes(&self, node_id: &Uuid) -> &Vec<Uuid> {
        &self.nodes.get(node_id).unwrap().next_nodes
    }
}
