export interface Position {
  x: number;
  y: number;
}

export enum CellType {
  EMPTY = 0,
  WALL = 1,
  START = 2,
  END = 3,
  PATH = 4,
  VISITED = 5,
  CURRENT = 6
}

export interface MazeCell {
  type: CellType;
  visited: boolean;
  distance: number;
  parent?: Position;
  gCost?: number;  // For A* algorithm
  hCost?: number;  // For A* algorithm
  fCost?: number;  // For A* algorithm
}

export interface Maze {
  width: number;
  height: number;
  grid: MazeCell[][];
  start: Position;
  end: Position;
}

export enum AlgorithmType {
  BFS = 'Breadth-First Search',
  DFS = 'Depth-First Search',
  DIJKSTRA = 'Dijkstra',
  ASTAR = 'A*',
  QLEARNING = 'Q-Learning',
  DQN = 'Deep Q-Network'
}

export interface AlgorithmStep {
  current: Position;
  visited: Position[];
  path: Position[];
  completed: boolean;
  message: string;
}

export interface AlgorithmConfig {
  speed: number; // milliseconds between steps
  showVisited: boolean;
  showPath: boolean;
  diagonalMovement: boolean;
} 