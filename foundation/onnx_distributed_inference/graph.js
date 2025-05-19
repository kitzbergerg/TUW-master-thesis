import { v4 as uuidv4 } from 'uuid';

export class Node {
  constructor(name, next, data) {
    this.id = uuidv4();
    this.name = name;
    this.next = next;
    this.data = data;
  }
}

export class Graph {
  constructor(nodes) {
    const nodesWithIds = nodes.map(node => ([node.id, node]))
    this.startNodeId = nodes[0].id;
    this.nodes = new Map(nodesWithIds)
    this.assignedWorkers = new Map(this.nodes.keys().map(id => ([id, []])))
  }

  getNextNodes(nodeId) {
    return this.nodes.get(nodeId).next
  }

  notYetAssignedNodes() {
    return [...this.assignedWorkers.entries().filter((arr) => arr[1].length == 0).map((arr) => arr[0])];
  }

  assignNode(nodeId, userId) {
    this.assignedWorkers.get(nodeId).push(userId)
  }

  getWorker(nodeId) {
    const assignedWorkers = this.assignedWorkers.get(nodeId);
    if (assignedWorkers.length == 0) {
      console.log("No available workers")
      return null;
    }
    return assignedWorkers[0]
  }

  addUser(userId) {
    const notAssignedYet = this.notYetAssignedNodes();
    // TODO: store all available users so that on disconnect another one can be assigned a node
    if (notAssignedYet.size == 0) return
    this.assignNode(notAssignedYet[0], userId)
  }

  removeUser(userId) {
    this.assignedWorkers.entries().forEach((arr) => {
      const users = arr[1];
      const index = users.indexOf(userId)
      if (index > -1) users.splice(index, 1)
    })
  }
}
