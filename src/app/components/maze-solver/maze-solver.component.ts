import { Component, OnInit, OnDestroy } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { MatCardModule } from '@angular/material/card';
import { MatButtonModule } from '@angular/material/button';
import { MatIconModule } from '@angular/material/icon';
import { MatSelectModule } from '@angular/material/select';
import { MatFormFieldModule } from '@angular/material/form-field';
import { MatSlider, MatSliderModule } from '@angular/material/slider';
import { MatCheckboxModule } from '@angular/material/checkbox';
import { MatProgressBarModule } from '@angular/material/progress-bar';
import { MatTooltipModule } from '@angular/material/tooltip';
import { Subscription } from 'rxjs';

import { MazeService } from '../../services/maze.service';
import { 
  Maze, 
  AlgorithmType, 
  AlgorithmConfig, 
  AlgorithmStep, 
  CellType 
} from '../../models/maze.model';

@Component({
  selector: 'app-maze-solver',
  standalone: true,
  imports: [
    CommonModule,
    FormsModule,
    MatCardModule,
    MatButtonModule,
    MatIconModule,
    MatSelectModule,
    MatFormFieldModule,
    MatSlider,
    MatSliderModule,
    MatCheckboxModule,
    MatProgressBarModule,
    MatTooltipModule
  ],
  templateUrl: './maze-solver.component.html',
  styleUrl: './maze-solver.component.css'
})
export class MazeSolverComponent implements OnInit, OnDestroy {
  maze: Maze | null = null;
  algorithmStep: AlgorithmStep | null = null;
  isRunning = false;
  
  selectedAlgorithm: AlgorithmType = AlgorithmType.BFS;
  algorithms = [
    { value: AlgorithmType.BFS, name: 'Breadth-First Search (BFS)', description: 'Explores all neighbors at current depth before moving deeper' },
    { value: AlgorithmType.DFS, name: 'Depth-First Search (DFS)', description: 'Explores as far as possible along each branch before backtracking' },
    { value: AlgorithmType.DIJKSTRA, name: 'Dijkstra\'s Algorithm', description: 'Finds shortest path by exploring nodes with lowest cost first' },
    { value: AlgorithmType.ASTAR, name: 'A* Algorithm', description: 'Uses heuristic to guide search toward the goal more efficiently' }
  ];

  config: AlgorithmConfig = {
    speed: 100,
    showVisited: true,
    showPath: true,
    diagonalMovement: false
  };

  mazeSize = { width: 21, height: 15 };
  CellType = CellType;

  private subscriptions: Subscription[] = [];

  constructor(private mazeService: MazeService) {}

  ngOnInit(): void {
    this.subscriptions.push(
      this.mazeService.currentMaze$.subscribe(maze => {
        this.maze = maze;
      }),
      this.mazeService.algorithmStep$.subscribe(step => {
        this.algorithmStep = step;
      }),
      this.mazeService.algorithmRunning$.subscribe(running => {
        this.isRunning = running;
      })
    );

    this.generateSimpleMaze();
  }

  ngOnDestroy(): void {
    this.subscriptions.forEach(sub => sub.unsubscribe());
    this.mazeService.stopAlgorithm();
  }

  generateMaze(): void {
    this.mazeService.stopAlgorithm();
    this.mazeService.generateMaze(this.mazeSize.width, this.mazeSize.height);
  }

  generateSimpleMaze(): void {
    this.mazeService.stopAlgorithm();
    this.mazeService.generateSimpleMaze(this.mazeSize.width, this.mazeSize.height);
  }

  solveMaze(): void {
    if (!this.maze) return;
    this.mazeService.solveMaze(this.selectedAlgorithm, this.config);
  }

  stopSolving(): void {
    this.mazeService.stopAlgorithm();
  }

  onCellClick(x: number, y: number): void {
    if (!this.isRunning) {
      this.mazeService.toggleCell(x, y);
    }
  }

  getCellClass(cell: any): string {
    const classes = ['maze-cell'];
    
    switch (cell.type) {
      case CellType.WALL:
        classes.push('wall');
        break;
      case CellType.START:
        classes.push('start');
        break;
      case CellType.END:
        classes.push('end');
        break;
      case CellType.PATH:
        classes.push('path');
        break;
      case CellType.VISITED:
        classes.push('visited');
        break;
      case CellType.CURRENT:
        classes.push('current');
        break;
      default:
        classes.push('empty');
    }

    return classes.join(' ');
  }

  getAlgorithmDescription(): string {
    const algorithm = this.algorithms.find(a => a.value === this.selectedAlgorithm);
    return algorithm ? algorithm.description : '';
  }

  onSpeedChange(event: Event): void {
    const value = (event.target as HTMLInputElement).value;
    if (value !== null) {
      this.config.speed = 1000 - (Number(value) * 10); // Convert to delay in ms
    }
  }

  getSpeedValue(): string {
    return ((1000 - this.config.speed) / 10).toString();
  }
} 