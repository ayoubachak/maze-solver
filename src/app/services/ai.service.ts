import { Injectable, NgZone } from '@angular/core';
import { BehaviorSubject, Observable, Subject, interval, Subscription } from 'rxjs';
import * as tf from '@tensorflow/tfjs';

import { Maze, Position, CellType, AlgorithmType } from '../models/maze.model';
import { 
  Action, 
  State, 
  TrainingConfig, 
  TrainingStats, 
  NeuralNetworkConfig, 
  Experience 
} from '../models/ai.model';

interface QTable {
  [stateKey: string]: number[]; // qValues for each action
}

@Injectable({
  providedIn: 'root'
})
export class AiService {
  private maze: Maze | null = null;
  private qTable: QTable = {};
  private dqnModel: tf.Sequential | null = null;
  private targetDqnModel: tf.Sequential | null = null;
  private replayMemory: Experience[] = [];

  private trainingStatusSubject = new BehaviorSubject<{ isRunning: boolean, message?: string }>({ isRunning: false });
  private trainingStatsSubject = new BehaviorSubject<TrainingStats | null>(null);
  private agentMovedSubject = new Subject<Position>();
  private trainingSubscription?: Subscription;

  trainingStatus$ = this.trainingStatusSubject.asObservable();
  trainingStats$ = this.trainingStatsSubject.asObservable();
  agentMoved$ = this.agentMovedSubject.asObservable();

  private currentConfig: TrainingConfig | null = null;
  private currentNnConfig: NeuralNetworkConfig | null = null;
  private currentAlgorithm: AlgorithmType | null = null;
  private currentEpisode = 0;
  private totalRewardsHistory: number[] = [];
  private successHistory: boolean[] = [];

  constructor(private ngZone: NgZone) {}

  initializeEnvironment(maze: Maze): void {
    this.maze = maze;
    this.qTable = {}; // Reset Q-table for new maze
    this.dqnModel = null; // Reset DQN model
    this.targetDqnModel = null;
    this.replayMemory = [];
    console.log('AI Environment Initialized with new maze', maze);
  }

  startTraining(
    algorithm: AlgorithmType, 
    config: TrainingConfig, 
    nnConfig: NeuralNetworkConfig
  ): void {
    if (!this.maze) {
      console.error('Maze not initialized for AI training');
      return;
    }
    this.stopTraining(); // Stop any ongoing training

    this.currentAlgorithm = algorithm;
    this.currentConfig = config;
    this.currentNnConfig = nnConfig;
    this.currentEpisode = 0;
    this.totalRewardsHistory = [];
    this.successHistory = [];

    this.trainingStatusSubject.next({ isRunning: true });

    if (algorithm === AlgorithmType.DQN) {
      this.initializeDqnModel();
    }

    // Run training outside Angular zone to prevent excessive change detection
    this.ngZone.runOutsideAngular(() => {
      this.trainingSubscription = interval(10) // Small delay for visualization updates
        .subscribe(() => {
          if (this.currentEpisode < this.currentConfig!.episodes) {
            this.runEpisode();
          } else {
            this.stopTraining('Training completed: All episodes finished.');
          }
        });
    });
  }

  private initializeDqnModel(): void {
    if (!this.currentNnConfig || !this.maze) return;

    const inputShape = this.getStateRepresentation(this.maze.start).length;
    const numActions = Object.keys(Action).length / 2; // Enum has string keys too

    this.dqnModel = this.createDqnModel(inputShape, numActions, this.currentNnConfig);
    this.targetDqnModel = this.createDqnModel(inputShape, numActions, this.currentNnConfig);
    this.updateTargetModel();
    console.log('DQN Model Initialized');
  }

  private createDqnModel(inputShape: number, numActions: number, nnConfig: NeuralNetworkConfig): tf.Sequential {
    const model = tf.sequential();
    model.add(tf.layers.dense({ inputShape: [inputShape], units: nnConfig.hiddenLayers[0], activation: nnConfig.activation as any }));
    for (let i = 1; i < nnConfig.hiddenLayers.length; i++) {
      model.add(tf.layers.dense({ units: nnConfig.hiddenLayers[i], activation: nnConfig.activation as any }));
    }
    model.add(tf.layers.dense({ units: numActions, activation: 'linear' })); // Output layer for Q-values
    
    model.compile({ optimizer: nnConfig.optimizer, loss: 'meanSquaredError' });
    return model;
  }

  private updateTargetModel(): void {
    if (this.dqnModel && this.targetDqnModel) {
      this.targetDqnModel.setWeights(this.dqnModel.getWeights());
      console.log('Target DQN model updated.');
    }
  }

  private runEpisode(): void {
    if (!this.maze || !this.currentConfig) return;

    let currentPosition = { ...this.maze.start };
    let totalReward = 0;
    let steps = 0;
    let episodeSuccess = false;

    for (steps = 0; steps < this.currentConfig.maxStepsPerEpisode; steps++) {
      this.agentMovedSubject.next(currentPosition);
      
      const state = this.getCurrentState(currentPosition);
      const action = this.chooseAction(state);
      const { nextPosition, reward, done } = this.takeAction(currentPosition, action);
      const nextState = this.getCurrentState(nextPosition);
      
      totalReward += reward;

      if (this.currentAlgorithm === AlgorithmType.QLEARNING) {
        this.updateQTable(state, action, reward, nextState);
      } else if (this.currentAlgorithm === AlgorithmType.DQN) {
        this.storeExperience(state, action, reward, nextState, done);
        this.trainDqnModel();
      }

      currentPosition = nextPosition;

      if (done) {
        episodeSuccess = (currentPosition.x === this.maze.end.x && currentPosition.y === this.maze.end.y);
        break;
      }
    }
    
    this.agentMovedSubject.next(currentPosition); // Final position

    // Decay exploration rate for Q-Learning
    if (this.currentAlgorithm === AlgorithmType.QLEARNING && this.currentConfig.explorationRate > this.currentConfig.minExplorationRate) {
      this.currentConfig.explorationRate *= this.currentConfig.explorationDecay;
    }
    // For DQN, exploration decay might be handled differently or be part of a more complex strategy

    this.totalRewardsHistory.push(totalReward);
    this.successHistory.push(episodeSuccess);
    this.currentEpisode++;

    const stats: TrainingStats = {
      episode: this.currentEpisode,
      totalReward,
      steps,
      explorationRate: this.currentConfig.explorationRate, 
      success: episodeSuccess,
      averageReward: this.totalRewardsHistory.reduce((a, b) => a + b, 0) / this.totalRewardsHistory.length,
      successRate: (this.successHistory.filter(s => s).length / this.successHistory.length) * 100
    };
    this.trainingStatsSubject.next(stats);

    if (this.currentAlgorithm === AlgorithmType.DQN && this.currentNnConfig && this.currentEpisode % this.currentNnConfig.targetUpdateFrequency === 0) {
        this.updateTargetModel();
    }
  }

  private getCurrentState(position: Position): State {
    if (!this.maze) throw new Error("Maze not available for getCurrentState");
    
    const walls = [
      this.isWall(position.x, position.y - 1), // Up
      this.isWall(position.x, position.y + 1), // Down
      this.isWall(position.x - 1, position.y), // Left
      this.isWall(position.x + 1, position.y)  // Right
    ];
    const distanceToGoal = Math.abs(position.x - this.maze.end.x) + Math.abs(position.y - this.maze.end.y);
    return { position, walls, distanceToGoal };
  }
  
  private isWall(x: number, y: number): boolean {
    if (!this.maze || x < 0 || x >= this.maze.width || y < 0 || y >= this.maze.height) {
      return true; // Treat out-of-bounds as walls
    }
    return this.maze.grid[y][x].type === CellType.WALL;
  }

  private chooseAction(state: State): Action {
    if (!this.currentConfig) throw new Error("Config not available");

    if (Math.random() < this.currentConfig.explorationRate) {
      return Math.floor(Math.random() * 4) as Action; // Explore: random action
    } else {
      // Exploit: choose best action from Q-table or DQN model
      if (this.currentAlgorithm === AlgorithmType.QLEARNING) {
        const stateKey = this.getStateKey(state);
        const qValues = this.qTable[stateKey] || [0, 0, 0, 0];
        return qValues.indexOf(Math.max(...qValues)) as Action;
      } else if (this.currentAlgorithm === AlgorithmType.DQN && this.dqnModel) {
        const stateTensor = tf.tensor2d([this.getStateRepresentation(state.position)]);
        const qValuesTensor = this.dqnModel.predict(stateTensor) as tf.Tensor;
        const qValues = qValuesTensor.dataSync();
        tf.dispose([stateTensor, qValuesTensor]);
        return qValues.indexOf(Math.max(...qValues)) as Action;
      } else {
        return Math.floor(Math.random() * 4) as Action; // Fallback
      }
    }
  }

  private takeAction(currentPosition: Position, action: Action): { nextPosition: Position, reward: number, done: boolean } {
    if (!this.maze) throw new Error("Maze not available for takeAction");
    let { x, y } = currentPosition;

    switch (action) {
      case Action.UP:    y--; break;
      case Action.DOWN:  y++; break;
      case Action.LEFT:  x--; break;
      case Action.RIGHT: x++; break;
    }

    let reward = -1; // Small penalty for each step
    let done = false;

    if (this.isWall(x, y)) {
      reward = -10; // Penalty for hitting a wall
      x = currentPosition.x; // Stay in place
      y = currentPosition.y;
    } else if (x === this.maze.end.x && y === this.maze.end.y) {
      reward = 100; // Reward for reaching the end
      done = true;
    } else if (this.maze.grid[y][x].type === CellType.VISITED) {
        // reward = -2; // Slightly higher penalty for re-visiting, could be an option
    }

    return { nextPosition: { x, y }, reward, done };
  }

  // Q-Learning Specific Methods
  private updateQTable(state: State, action: Action, reward: number, nextState: State): void {
    if (!this.currentConfig) return;

    const stateKey = this.getStateKey(state);
    const nextStateKey = this.getStateKey(nextState);

    if (!this.qTable[stateKey]) this.qTable[stateKey] = [0, 0, 0, 0];
    if (!this.qTable[nextStateKey]) this.qTable[nextStateKey] = [0, 0, 0, 0];

    const oldQValue = this.qTable[stateKey][action];
    const nextMaxQ = Math.max(...this.qTable[nextStateKey]);
    
    const newQValue = oldQValue + this.currentConfig.learningRate * 
                      (reward + this.currentConfig.discountFactor * nextMaxQ - oldQValue);
    this.qTable[stateKey][action] = newQValue;
  }

  private getStateKey(state: State): string {
    // Simple state key: position + wall configuration
    return `${state.position.x}-${state.position.y}-${state.walls.map(w => w ? 1:0).join('')}`;
  }

  // DQN Specific Methods
  private getStateRepresentation(position: Position): number[] {
    if (!this.maze) return [];
    // Flattened grid representation (normalized) or more sophisticated features
    // For simplicity, using agent's (x,y) and (dist_x, dist_y) to goal, and wall booleans
    const stateArray: number[] = [];
    stateArray.push(position.x / this.maze.width);
    stateArray.push(position.y / this.maze.height);
    stateArray.push((this.maze.end.x - position.x) / this.maze.width);
    stateArray.push((this.maze.end.y - position.y) / this.maze.height);
    
    const s = this.getCurrentState(position);
    s.walls.forEach(wall => stateArray.push(wall ? 1 : 0));
    
    return stateArray;
  }

  private storeExperience(state: State, action: Action, reward: number, nextState: State, done: boolean): void {
    if (!this.currentNnConfig) return;
    if (this.replayMemory.length >= this.currentNnConfig.memorySize) {
      this.replayMemory.shift(); // Remove oldest experience
    }
    this.replayMemory.push({
      state: this.getStateRepresentation(state.position),
      action,
      reward,
      nextState: this.getStateRepresentation(nextState.position),
      done
    });
  }

  private async trainDqnModel(): Promise<void> {
    if (!this.dqnModel || !this.targetDqnModel || !this.currentNnConfig || this.replayMemory.length < this.currentNnConfig.batchSize) {
      return;
    }

    // Sample a minibatch from replay memory
    const batch = tf.tidy(() => {
        const samples: Experience[] = [];
        for (let i = 0; i < this.currentNnConfig!.batchSize; i++) {
          samples.push(this.replayMemory[Math.floor(Math.random() * this.replayMemory.length)]);
        }

        const states = samples.map(s => s.state);
        const actions = samples.map(s => s.action);
        const rewards = samples.map(s => s.reward);
        const nextStates = samples.map(s => s.nextState);
        const dones = samples.map(s => s.done);

        const currentQValues = this.dqnModel!.predict(tf.tensor2d(states)) as tf.Tensor;
        const nextQValuesFromTarget = this.targetDqnModel!.predict(tf.tensor2d(nextStates)) as tf.Tensor;
        const nextMaxQ = nextQValuesFromTarget.max(1);

        const targetQValuesData = currentQValues.arraySync() as number[][];

        for (let i = 0; i < samples.length; i++) {
          if (dones[i]) {
            targetQValuesData[i][actions[i]] = rewards[i];
          } else {
            targetQValuesData[i][actions[i]] = rewards[i] + this.currentConfig!.discountFactor * nextMaxQ.dataSync()[i];
          }
        }
        return { states: tf.tensor2d(states), targets: tf.tensor2d(targetQValuesData) };
    });
    
    const history = await this.dqnModel.fit(batch.states, batch.targets, {
        epochs: 1,
        verbose: 0
    });

    // Only dispose tensors, not the history object
    tf.dispose([batch.states, batch.targets]);
  }

  stopTraining(message?: string): void {
    if (this.trainingSubscription) {
      this.trainingSubscription.unsubscribe();
      this.trainingSubscription = undefined;
    }
    this.trainingStatusSubject.next({ isRunning: false, message: message || 'Training stopped by user.' });
    console.log(message || 'AI Training Stopped');
  }
  
  getQTableArray(): {state: string, qValues: number[]}[] {
      return Object.entries(this.qTable).map(([state, qValues]) => ({state, qValues}));
  }
} 