import { v4 as uuidv4 } from 'uuid';

export class Node {
  id: string;
  next: Node[];
  name: string;
  data: any;
  constructor(name: string, next: Node[], data: any) {
    this.id = uuidv4();
    this.name = name;
    this.next = next;
    this.data = data;
  }
}

export class Graph {
  startNodeId: string;
  nodes: Map<string, Node>;
  assignedWorkers: Map<string, string[]>;
  constructor(nodes: Node[]) {
    this.startNodeId = nodes[0].id;
    this.nodes = new Map(nodes.map(node => ([node.id, node])));
    this.assignedWorkers = new Map(Array.from(this.nodes.keys()).map(id => ([id, []])));
  }

  getNode(nodeId: string): Node | undefined {
    return this.nodes.get(nodeId);
  }

  getNextNodes(nodeId: string): Node[] | undefined {
    return this.nodes.get(nodeId)?.next;
  }

  notYetAssignedNodes(): string[] {
    return Array.from(this.assignedWorkers.entries()).filter((arr) => arr[1].length == 0).map((arr) => arr[0]);
  }

  assignNode(nodeId: string, userId: string) {
    this.assignedWorkers.get(nodeId)?.push(userId);
  }

  getWorker(nodeId: string) {
    const assignedWorkers = this.assignedWorkers.get(nodeId);
    if (assignedWorkers == undefined) {
      console.error("No such node");
      return null;
    }
    if (assignedWorkers.length == 0) {
      console.error("No available workers");
      return null;
    }
    return assignedWorkers[0];
  }

  addUser(userId: string): string | undefined {
    const notAssignedYet = this.notYetAssignedNodes();
    // TODO: store all available users so that on disconnect another one can be assigned a node
    if (notAssignedYet.length == 0) return;
    const nodeToBeAssigned = notAssignedYet[0];
    this.assignNode(nodeToBeAssigned, userId);
    return nodeToBeAssigned;
  }

  removeUser(userId: string) {
    Array.from(this.assignedWorkers.entries()).forEach((arr) => {
      const users = arr[1];
      const index = users.indexOf(userId);
      if (index > -1) users.splice(index, 1);
    })
  }
}
