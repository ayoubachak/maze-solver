import { Injectable } from '@angular/core';
import { BehaviorSubject, Observable, interval, Subscription } from 'rxjs';
import { 
  Maze, 
  MazeCell, 
  Position, 
  CellType, 
  AlgorithmType, 
  AlgorithmStep, 
  AlgorithmConfig 
} from '../models/maze.model';

@Injectable({
  providedIn: 'root'
})
export class MazeService {
  private currentMazeSubject = new BehaviorSubject<Maze | null>(null);
  private algorithmStepSubject = new BehaviorSubject<AlgorithmStep | null>(null);
  private algorithmRunningSubject = new BehaviorSubject<boolean>(false);

  currentMaze$ = this.currentMazeSubject.asObservable();
  algorithmStep$ = this.algorithmStepSubject.asObservable();
  algorithmRunning$ = this.algorithmRunningSubject.asObservable();

  private algorithmSubscription?: Subscription;

  constructor() {}

  generateMaze(width: number, height: number): Maze {
    const maze: Maze = {
      width,
      height,
      grid: [],
      start: { x: 1, y: 1 },
      end: { x: width - 2, y: height - 2 }
    };

    // Initialize grid with walls
    for (let y = 0; y < height; y++) {
      maze.grid[y] = [];
      for (let x = 0; x < width; x++) {
        maze.grid[y][x] = {
          type: CellType.WALL,
          visited: false,
          distance: Infinity
        };
      }
    }

    // Generate maze using recursive backtracking
    this.generateMazeRecursive(maze, maze.start.x, maze.start.y);

    // Set start and end points
    maze.grid[maze.start.y][maze.start.x].type = CellType.START;
    maze.grid[maze.end.y][maze.end.x].type = CellType.END;

    // Ensure end is reachable
    this.ensurePathExists(maze);

    this.currentMazeSubject.next(maze);
    return maze;
  }

  private generateMazeRecursive(maze: Maze, x: number, y: number): void {
    maze.grid[y][x].type = CellType.EMPTY;
    
    const directions = [
      { x: 0, y: -2 }, // Up
      { x: 2, y: 0 },  // Right
      { x: 0, y: 2 },  // Down
      { x: -2, y: 0 }  // Left
    ];

    // Randomize directions
    this.shuffleArray(directions);

    for (const dir of directions) {
      const newX = x + dir.x;
      const newY = y + dir.y;

      if (this.isValidCell(maze, newX, newY) && 
          maze.grid[newY][newX].type === CellType.WALL) {
        
        // Remove wall between current and new cell
        const wallX = x + dir.x / 2;
        const wallY = y + dir.y / 2;
        maze.grid[wallY][wallX].type = CellType.EMPTY;
        
        this.generateMazeRecursive(maze, newX, newY);
      }
    }
  }

  private ensurePathExists(maze: Maze): void {
    // Simple path from start to end
    let current = { ...maze.start };
    
    while (current.x !== maze.end.x || current.y !== maze.end.y) {
      if (current.x < maze.end.x) current.x++;
      else if (current.x > maze.end.x) current.x--;
      else if (current.y < maze.end.y) current.y++;
      else if (current.y > maze.end.y) current.y--;
      
      if (maze.grid[current.y][current.x].type === CellType.WALL) {
        maze.grid[current.y][current.x].type = CellType.EMPTY;
      }
    }
  }

  generateSimpleMaze(width: number, height: number): Maze {
    const maze: Maze = {
      width,
      height,
      grid: [],
      start: { x: 0, y: 0 },
      end: { x: width - 1, y: height - 1 }
    };

    // Create simple maze with some walls
    for (let y = 0; y < height; y++) {
      maze.grid[y] = [];
      for (let x = 0; x < width; x++) {
        const isWall = Math.random() < 0.3 && 
                      !(x === maze.start.x && y === maze.start.y) &&
                      !(x === maze.end.x && y === maze.end.y);
        
        maze.grid[y][x] = {
          type: isWall ? CellType.WALL : CellType.EMPTY,
          visited: false,
          distance: Infinity
        };
      }
    }

    maze.grid[maze.start.y][maze.start.x].type = CellType.START;
    maze.grid[maze.end.y][maze.end.x].type = CellType.END;

    this.currentMazeSubject.next(maze);
    return maze;
  }

  solveMaze(algorithm: AlgorithmType, config: AlgorithmConfig): void {
    const maze = this.currentMazeSubject.value;
    if (!maze) return;

    this.stopAlgorithm();
    this.resetMazeVisualization(maze);

    switch (algorithm) {
      case AlgorithmType.BFS:
        this.solveBFS(maze, config);
        break;
      case AlgorithmType.DFS:
        this.solveDFS(maze, config);
        break;
      case AlgorithmType.DIJKSTRA:
        this.solveDijkstra(maze, config);
        break;
      case AlgorithmType.ASTAR:
        this.solveAStar(maze, config);
        break;
    }
  }

  private solveBFS(maze: Maze, config: AlgorithmConfig): void {
    const queue: Position[] = [maze.start];
    const visited = new Set<string>();
    const parent = new Map<string, Position>();
    
    this.algorithmRunningSubject.next(true);
    
    this.algorithmSubscription = interval(config.speed).subscribe(() => {
      if (queue.length === 0) {
        this.completeAlgorithm(false, "No path found");
        return;
      }

      const current = queue.shift()!;
      const currentKey = `${current.x},${current.y}`;
      
      if (visited.has(currentKey)) return;
      visited.add(currentKey);

      if (maze.grid[current.y][current.x].type !== CellType.START) {
        maze.grid[current.y][current.x].type = CellType.VISITED;
      }

      if (current.x === maze.end.x && current.y === maze.end.y) {
        this.reconstructPath(maze, parent, current);
        this.completeAlgorithm(true, "Path found using BFS!");
        return;
      }

      const neighbors = this.getNeighbors(maze, current, config.diagonalMovement);
      for (const neighbor of neighbors) {
        const neighborKey = `${neighbor.x},${neighbor.y}`;
        if (!visited.has(neighborKey)) {
          queue.push(neighbor);
          parent.set(neighborKey, current);
        }
      }

      this.algorithmStepSubject.next({
        current,
        visited: Array.from(visited).map(key => {
          const [x, y] = key.split(',').map(Number);
          return { x, y };
        }),
        path: [],
        completed: false,
        message: `Exploring: (${current.x}, ${current.y})`
      });

      this.currentMazeSubject.next({ ...maze });
    });
  }

  private solveDFS(maze: Maze, config: AlgorithmConfig): void {
    const stack: Position[] = [maze.start];
    const visited = new Set<string>();
    const parent = new Map<string, Position>();
    
    this.algorithmRunningSubject.next(true);
    
    this.algorithmSubscription = interval(config.speed).subscribe(() => {
      if (stack.length === 0) {
        this.completeAlgorithm(false, "No path found");
        return;
      }

      const current = stack.pop()!;
      const currentKey = `${current.x},${current.y}`;
      
      if (visited.has(currentKey)) return;
      visited.add(currentKey);

      if (maze.grid[current.y][current.x].type !== CellType.START) {
        maze.grid[current.y][current.x].type = CellType.VISITED;
      }

      if (current.x === maze.end.x && current.y === maze.end.y) {
        this.reconstructPath(maze, parent, current);
        this.completeAlgorithm(true, "Path found using DFS!");
        return;
      }

      const neighbors = this.getNeighbors(maze, current, config.diagonalMovement);
      for (const neighbor of neighbors) {
        const neighborKey = `${neighbor.x},${neighbor.y}`;
        if (!visited.has(neighborKey)) {
          stack.push(neighbor);
          parent.set(neighborKey, current);
        }
      }

      this.algorithmStepSubject.next({
        current,
        visited: Array.from(visited).map(key => {
          const [x, y] = key.split(',').map(Number);
          return { x, y };
        }),
        path: [],
        completed: false,
        message: `Exploring: (${current.x}, ${current.y})`
      });

      this.currentMazeSubject.next({ ...maze });
    });
  }

  private solveAStar(maze: Maze, config: AlgorithmConfig): void {
    const openSet: Position[] = [maze.start];
    const closedSet = new Set<string>();
    const parent = new Map<string, Position>();
    
    // Initialize costs
    for (let y = 0; y < maze.height; y++) {
      for (let x = 0; x < maze.width; x++) {
        maze.grid[y][x].gCost = Infinity;
        maze.grid[y][x].hCost = this.heuristic({ x, y }, maze.end);
        maze.grid[y][x].fCost = Infinity;
      }
    }
    
    maze.grid[maze.start.y][maze.start.x].gCost = 0;
    maze.grid[maze.start.y][maze.start.x].fCost = maze.grid[maze.start.y][maze.start.x].hCost!;
    
    this.algorithmRunningSubject.next(true);
    
    this.algorithmSubscription = interval(config.speed).subscribe(() => {
      if (openSet.length === 0) {
        this.completeAlgorithm(false, "No path found");
        return;
      }

      // Find node with lowest fCost
      let currentIndex = 0;
      for (let i = 1; i < openSet.length; i++) {
        const current = openSet[i];
        const lowest = openSet[currentIndex];
        if (maze.grid[current.y][current.x].fCost! < maze.grid[lowest.y][lowest.x].fCost!) {
          currentIndex = i;
        }
      }

      const current = openSet.splice(currentIndex, 1)[0];
      const currentKey = `${current.x},${current.y}`;
      closedSet.add(currentKey);

      if (maze.grid[current.y][current.x].type !== CellType.START) {
        maze.grid[current.y][current.x].type = CellType.VISITED;
      }

      if (current.x === maze.end.x && current.y === maze.end.y) {
        this.reconstructPath(maze, parent, current);
        this.completeAlgorithm(true, "Path found using A*!");
        return;
      }

      const neighbors = this.getNeighbors(maze, current, config.diagonalMovement);
      for (const neighbor of neighbors) {
        const neighborKey = `${neighbor.x},${neighbor.y}`;
        if (closedSet.has(neighborKey)) continue;

        const tentativeGCost = maze.grid[current.y][current.x].gCost! + 1;
        
        if (tentativeGCost < maze.grid[neighbor.y][neighbor.x].gCost!) {
          parent.set(neighborKey, current);
          maze.grid[neighbor.y][neighbor.x].gCost = tentativeGCost;
          maze.grid[neighbor.y][neighbor.x].fCost = tentativeGCost + maze.grid[neighbor.y][neighbor.x].hCost!;
          
          if (!openSet.find(pos => pos.x === neighbor.x && pos.y === neighbor.y)) {
            openSet.push(neighbor);
          }
        }
      }

      this.algorithmStepSubject.next({
        current,
        visited: Array.from(closedSet).map(key => {
          const [x, y] = key.split(',').map(Number);
          return { x, y };
        }),
        path: [],
        completed: false,
        message: `Exploring: (${current.x}, ${current.y}) - f: ${maze.grid[current.y][current.x].fCost?.toFixed(1)}`
      });

      this.currentMazeSubject.next({ ...maze });
    });
  }

  private solveDijkstra(maze: Maze, config: AlgorithmConfig): void {
    const distances: number[][] = Array(maze.height).fill(null).map(() => Array(maze.width).fill(Infinity));
    const visited = new Set<string>();
    const parent = new Map<string, Position>();
    const priorityQueue: Position[] = [];

    distances[maze.start.y][maze.start.x] = 0;
    priorityQueue.push(maze.start);

    this.algorithmRunningSubject.next(true);

    this.algorithmSubscription = interval(config.speed).subscribe(() => {
      if (priorityQueue.length === 0) {
        this.completeAlgorithm(false, "No path found");
        return;
      }

      // Find node with minimum distance
      let currentIndex = 0;
      for (let i = 1; i < priorityQueue.length; i++) {
        const current = priorityQueue[i];
        const lowest = priorityQueue[currentIndex];
        if (distances[current.y][current.x] < distances[lowest.y][lowest.x]) {
          currentIndex = i;
        }
      }

      const current = priorityQueue.splice(currentIndex, 1)[0];
      const currentKey = `${current.x},${current.y}`;
      
      if (visited.has(currentKey)) return;
      visited.add(currentKey);

      if (maze.grid[current.y][current.x].type !== CellType.START) {
        maze.grid[current.y][current.x].type = CellType.VISITED;
      }

      if (current.x === maze.end.x && current.y === maze.end.y) {
        this.reconstructPath(maze, parent, current);
        this.completeAlgorithm(true, "Shortest path found using Dijkstra!");
        return;
      }

      const neighbors = this.getNeighbors(maze, current, config.diagonalMovement);
      for (const neighbor of neighbors) {
        const neighborKey = `${neighbor.x},${neighbor.y}`;
        if (visited.has(neighborKey)) continue;

        const newDist = distances[current.y][current.x] + 1;
        if (newDist < distances[neighbor.y][neighbor.x]) {
          distances[neighbor.y][neighbor.x] = newDist;
          parent.set(neighborKey, current);
          
          if (!priorityQueue.find(pos => pos.x === neighbor.x && pos.y === neighbor.y)) {
            priorityQueue.push(neighbor);
          }
        }
      }

      this.algorithmStepSubject.next({
        current,
        visited: Array.from(visited).map(key => {
          const [x, y] = key.split(',').map(Number);
          return { x, y };
        }),
        path: [],
        completed: false,
        message: `Distance: ${distances[current.y][current.x]} at (${current.x}, ${current.y})`
      });

      this.currentMazeSubject.next({ ...maze });
    });
  }

  private getNeighbors(maze: Maze, pos: Position, diagonal: boolean): Position[] {
    const neighbors: Position[] = [];
    const directions = diagonal 
      ? [
          { x: 0, y: -1 }, { x: 1, y: 0 }, { x: 0, y: 1 }, { x: -1, y: 0 },  // Cardinal
          { x: 1, y: -1 }, { x: 1, y: 1 }, { x: -1, y: 1 }, { x: -1, y: -1 }   // Diagonal
        ]
      : [{ x: 0, y: -1 }, { x: 1, y: 0 }, { x: 0, y: 1 }, { x: -1, y: 0 }];

    for (const dir of directions) {
      const newX = pos.x + dir.x;
      const newY = pos.y + dir.y;

      if (this.isValidCell(maze, newX, newY) && 
          maze.grid[newY][newX].type !== CellType.WALL) {
        neighbors.push({ x: newX, y: newY });
      }
    }

    return neighbors;
  }

  private heuristic(a: Position, b: Position): number {
    return Math.abs(a.x - b.x) + Math.abs(a.y - b.y); // Manhattan distance
  }

  private reconstructPath(maze: Maze, parent: Map<string, Position>, end: Position): void {
    const path: Position[] = [];
    let current: Position | undefined = end;

    while (current) {
      path.unshift(current);
      const currentKey = `${current.x},${current.y}`;
      current = parent.get(currentKey);
    }

    // Mark path
    for (const pos of path) {
      if (maze.grid[pos.y][pos.x].type !== CellType.START && 
          maze.grid[pos.y][pos.x].type !== CellType.END) {
        maze.grid[pos.y][pos.x].type = CellType.PATH;
      }
    }
  }

  private resetMazeVisualization(maze: Maze): void {
    for (let y = 0; y < maze.height; y++) {
      for (let x = 0; x < maze.width; x++) {
        if (maze.grid[y][x].type === CellType.VISITED || 
            maze.grid[y][x].type === CellType.PATH ||
            maze.grid[y][x].type === CellType.CURRENT) {
          maze.grid[y][x].type = CellType.EMPTY;
        }
        maze.grid[y][x].visited = false;
        maze.grid[y][x].distance = Infinity;
        maze.grid[y][x].gCost = undefined;
        maze.grid[y][x].hCost = undefined;
        maze.grid[y][x].fCost = undefined;
      }
    }

    maze.grid[maze.start.y][maze.start.x].type = CellType.START;
    maze.grid[maze.end.y][maze.end.x].type = CellType.END;
  }

  private completeAlgorithm(success: boolean, message: string): void {
    this.algorithmRunningSubject.next(false);
    this.algorithmStepSubject.next({
      current: { x: -1, y: -1 },
      visited: [],
      path: [],
      completed: true,
      message
    });
    
    if (this.algorithmSubscription) {
      this.algorithmSubscription.unsubscribe();
    }
  }

  stopAlgorithm(): void {
    if (this.algorithmSubscription) {
      this.algorithmSubscription.unsubscribe();
      this.algorithmRunningSubject.next(false);
    }
  }

  toggleCell(x: number, y: number): void {
    const maze = this.currentMazeSubject.value;
    if (!maze) return;

    const cell = maze.grid[y][x];
    if (cell.type === CellType.START || cell.type === CellType.END) return;

    cell.type = cell.type === CellType.WALL ? CellType.EMPTY : CellType.WALL;
    this.currentMazeSubject.next({ ...maze });
  }

  private isValidCell(maze: Maze, x: number, y: number): boolean {
    return x >= 0 && x < maze.width && y >= 0 && y < maze.height;
  }

  private shuffleArray<T>(array: T[]): void {
    for (let i = array.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [array[i], array[j]] = [array[j], array[i]];
    }
  }
} 