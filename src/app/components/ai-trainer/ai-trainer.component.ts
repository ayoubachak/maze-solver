import { Component, OnInit, OnDestroy, ChangeDetectorRef, NgZone } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { MatCardModule } from '@angular/material/card';
import { MatButtonModule } from '@angular/material/button';
import { MatIconModule } from '@angular/material/icon';
import { MatSelectModule } from '@angular/material/select';
import { MatFormFieldModule } from '@angular/material/form-field';
import { MatSliderModule } from '@angular/material/slider';
import { MatCheckboxModule } from '@angular/material/checkbox';
import { MatProgressBarModule } from '@angular/material/progress-bar';
import { MatTabsModule } from '@angular/material/tabs';
import { MatInputModule } from '@angular/material/input';
import { MatTooltipModule } from '@angular/material/tooltip';
import { Subscription } from 'rxjs';

import { MazeService } from '../../services/maze.service';
import { AiService } from '../../services/ai.service';
import { Maze, CellType, AlgorithmType } from '../../models/maze.model';
import { TrainingConfig, TrainingStats, NeuralNetworkConfig } from '../../models/ai.model';

@Component({
  selector: 'app-ai-trainer',
  standalone: true,
  imports: [
    CommonModule,
    FormsModule,
    MatCardModule,
    MatButtonModule,
    MatIconModule,
    MatSelectModule,
    MatFormFieldModule,
    MatSliderModule,
    MatCheckboxModule,
    MatProgressBarModule,
    MatTabsModule,
    MatInputModule,
    MatTooltipModule
  ],
  templateUrl: './ai-trainer.component.html',
  styleUrl: './ai-trainer.component.css'
})
export class AiTrainerComponent implements OnInit, OnDestroy {
  maze: Maze | null = null;
  isRunning = false;
  trainingStats: TrainingStats | null = null;
  currentAgentPosition: { x: number, y: number } | null = null;

  selectedAlgorithm: AlgorithmType = AlgorithmType.QLEARNING;
  algorithms = [
    { value: AlgorithmType.QLEARNING, name: 'Q-Learning', description: 'A model-free reinforcement learning algorithm to learn the quality of actions.' },
    { value: AlgorithmType.DQN, name: 'Deep Q-Network (DQN)', description: 'Uses a neural network to approximate Q-values, suitable for complex states.' }
  ];

  qLearningConfig: TrainingConfig = {
    learningRate: 0.1,
    discountFactor: 0.9,
    explorationRate: 1.0,
    explorationDecay: 0.995,
    minExplorationRate: 0.01,
    episodes: 1000,
    maxStepsPerEpisode: 200
  };

  dqnConfig: NeuralNetworkConfig = {
    hiddenLayers: [128, 128],
    activation: 'relu',
    optimizer: 'adam',
    learningRate: 0.001,
    batchSize: 32,
    memorySize: 10000,
    targetUpdateFrequency: 10
  };
  
  mazeSize = { width: 15, height: 11 }; // Smaller maze for faster training
  CellType = CellType;
  AlgorithmType = AlgorithmType;
  
  private subscriptions: Subscription[] = [];

  constructor(
    private mazeService: MazeService,
    private aiService: AiService,
    private cdr: ChangeDetectorRef,
    private ngZone: NgZone
  ) {}

  ngOnInit(): void {
    this.subscriptions.push(
      this.mazeService.currentMaze$.subscribe(maze => {
        this.maze = maze;
        if (maze) {
          this.aiService.initializeEnvironment(maze);
        }
        this.cdr.detectChanges();
      }),
      this.aiService.trainingStats$.subscribe(stats => {
        this.ngZone.run(() => {
          this.trainingStats = stats;
          this.cdr.detectChanges();
        });
      }),
      this.aiService.agentMoved$.subscribe(position => {
        this.ngZone.run(() => {
          this.currentAgentPosition = position;
          if (this.maze && position) {
            // Clear previous agent position
            this.maze.grid.forEach(row => row.forEach(cell => {
              if (cell.type === CellType.CURRENT) cell.type = CellType.EMPTY;
            }));
            // Mark new agent position
            if(this.maze.grid[position.y][position.x].type !== CellType.END && this.maze.grid[position.y][position.x].type !== CellType.START) {
                this.maze.grid[position.y][position.x].type = CellType.CURRENT;
            }
          }
          this.cdr.detectChanges(); 
        });
      }),
      this.aiService.trainingStatus$.subscribe(status => {
        this.isRunning = status.isRunning;
        if (!status.isRunning && status.message) {
            console.log("Training complete: ", status.message);
        }
        this.cdr.detectChanges();
      })
    );
    this.generateMaze(); 
  }

  ngOnDestroy(): void {
    this.subscriptions.forEach(sub => sub.unsubscribe());
    this.aiService.stopTraining();
  }

  generateMaze(): void {
    this.aiService.stopTraining();
    this.mazeService.generateMaze(this.mazeSize.width, this.mazeSize.height);
  }

  startTraining(): void {
    if (!this.maze) return;
    this.aiService.startTraining(this.selectedAlgorithm, this.qLearningConfig, this.dqnConfig);
  }

  stopTraining(): void {
    this.aiService.stopTraining();
  }

  getCellClass(cellType: CellType, x: number, y: number): string {
    const classes = ['maze-cell'];
    if (this.currentAgentPosition && this.currentAgentPosition.x === x && this.currentAgentPosition.y === y) {
      classes.push('current');
    } else {
      switch (cellType) {
        case CellType.WALL: classes.push('wall'); break;
        case CellType.START: classes.push('start'); break;
        case CellType.END: classes.push('end'); break;
        case CellType.PATH: classes.push('path'); break; // AI might draw its own path
        case CellType.VISITED: classes.push('visited'); break; // For AI exploration visualization
        default: classes.push('empty');
      }
    }
    return classes.join(' ');
  }

  getAlgorithmDescription(): string {
    return this.algorithms.find(a => a.value === this.selectedAlgorithm)?.description || '';
  }
  
  formatSliderLabel(value: number): string {
    if (value >= 1000) {
      return Math.round(value / 1000) + 'k';
    }
    return `${value}`;
  }
} 