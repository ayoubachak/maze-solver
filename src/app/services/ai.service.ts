import { Injectable, NgZone } from '@angular/core';
import { BehaviorSubject, Subject, interval, Subscription } from 'rxjs';
import * as tf from '@tensorflow/tfjs';
import { Sequential } from '@tensorflow/tfjs-layers';

import { Maze, Position, CellType, AlgorithmType } from '../models/maze.model';
import { 
  Action, 
  State, 
  TrainingConfig, 
  TrainingStats, 
  NeuralNetworkConfig, 
  Experience,
  VisualizationSettings
} from '../models/ai.model';

interface QTable {
  [stateKey: string]: number[]; // qValues for each action
}

// Define valid activation types
type ActivationType = 'relu' | 'sigmoid' | 'tanh' | 'linear';

// Update NeuralNetworkConfig interface to use correct activation type
interface NeuralNetworkConfigInternal extends Omit<NeuralNetworkConfig, 'activation'> {
  activation: ActivationType;
}

@Injectable({
  providedIn: 'root'
})
export class AiService {
  private maze: Maze | null = null;
  private qTable: QTable = {};
  private dqnModel: Sequential | null = null;
  private targetDqnModel: Sequential | null = null;
  private replayMemory: Experience[] = [];
  private currentNnConfig: NeuralNetworkConfigInternal | null = null;

  private readonly trainingStatusSubject = new BehaviorSubject<{ isRunning: boolean, isPaused: boolean, message?: string }>({ isRunning: false, isPaused: false });
  private readonly trainingStatsSubject = new BehaviorSubject<TrainingStats | null>(null);
  private readonly agentMovedSubject = new Subject<Position>();
  private trainingSubscription?: Subscription;

  // Add testing observables
  private readonly testingStatusSubject = new BehaviorSubject<{ isRunning: boolean, message?: string }>({ isRunning: false });
  private readonly testStatsSubject = new BehaviorSubject<{
    totalSteps: number,
    success: boolean,
    reward: number,
    path: Position[]
  } | null>(null);

  trainingStatus$ = this.trainingStatusSubject.asObservable();
  trainingStats$ = this.trainingStatsSubject.asObservable();
  agentMoved$ = this.agentMovedSubject.asObservable();
  testingStatus$ = this.testingStatusSubject.asObservable();
  testStats$ = this.testStatsSubject.asObservable();

  private currentConfig: TrainingConfig | null = null;
  private currentAlgorithm: AlgorithmType | null = null;
  private currentEpisode = 0;
  private totalRewardsHistory: number[] = [];
  private successHistory: boolean[] = [];
  private isPaused = false;

  // Enhanced visualization properties
  private readonly agentPathHistory: Position[] = [];
  private readonly exploredCells: Set<string> = new Set();
  private visualizationSettings: VisualizationSettings = {
    enabled: true,
    speed: 50,
    showPath: true,
    showExplorationHeatmap: true,
    showAgentTrail: true,
    performanceMode: false,
    adaptiveQuality: true,
    maxHistorySize: 100
  };
  private lastVisualizationUpdate = 0;
  private readonly visitedCells: Map<string, number> = new Map();
  private readonly explorationHeatmap: Map<string, number> = new Map();
  private visualizationHistory: { position: Position, timestamp: number }[] = [];

  // Performance optimization variables
  private lastStatsUpdate = 0;
  private readonly STATS_UPDATE_INTERVAL = 250; // Update stats max every 250ms
  private visualizationSpeed = 50; // Configurable visualization speed (1-100)
  private enableOptimizedVisualization = true;
  private frameSkipCount = 0;
  private lastVisualizationUpdateTime = 0;
  private readonly VISUALIZATION_UPDATE_INTERVAL = 100; // Update viz max every 100ms

  constructor(private readonly ngZone: NgZone) {}

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
    this.currentNnConfig = nnConfig as NeuralNetworkConfigInternal;
    this.currentEpisode = 0;
    this.totalRewardsHistory = [];
    this.successHistory = [];
    this.isPaused = false;

    this.trainingStatusSubject.next({ isRunning: true, isPaused: false });

    if (algorithm === AlgorithmType.DQN) {
      this.initializeDqnModel();
    }

    // Run training outside Angular zone to prevent excessive change detection
    this.ngZone.runOutsideAngular(() => {
      this.trainingSubscription = interval(10) // Small delay for visualization updates
        .subscribe(() => {
          if (!this.isPaused && this.currentEpisode < this.currentConfig!.episodes) {
            this.runEpisode();
          } else if (!this.isPaused && this.currentEpisode >= this.currentConfig!.episodes) {
            this.stopTraining('Training completed: All episodes finished.');
          }
        });
    });
  }

  pauseTraining(): void {
    if (this.trainingStatusSubject.value.isRunning) {
      this.isPaused = true;
      this.trainingStatusSubject.next({ isRunning: true, isPaused: true, message: 'Training paused' });
      console.log('Training paused at episode', this.currentEpisode);
    }
  }

  resumeTraining(): void {
    if (this.trainingStatusSubject.value.isRunning && this.isPaused) {
      this.isPaused = false;
      this.trainingStatusSubject.next({ isRunning: true, isPaused: false, message: 'Training resumed' });
      console.log('Training resumed from episode', this.currentEpisode);
    }
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

    // Clear visualization history for new episode if enabled
    if (this.visualizationSettings.enabled && this.currentEpisode === 0) {
      this.clearVisualizationHistory();
    }

    for (steps = 0; steps < this.currentConfig.maxStepsPerEpisode; steps++) {
      // Track agent movement for enhanced visualization
      this.trackAgentMovement(currentPosition);
      
      // Only update visualization if settings allow and it's time to update
      if (this.shouldUpdateVisualization()) {
        this.agentMovedSubject.next(currentPosition);
      }
      
      const state = this.getCurrentState(currentPosition);
      const action = this.chooseAction(state);
      const { nextPosition, reward, done } = this.takeAction(currentPosition, action);
      const nextState = this.getCurrentState(nextPosition);
      
      totalReward += reward;

      if (this.currentAlgorithm === AlgorithmType.QLEARNING) {
        this.updateQTable(state, action, reward, nextState);
      } else if (this.currentAlgorithm === AlgorithmType.DQN) {
        this.storeExperience(state, action, reward, nextState, done);
        if (steps % 4 === 0) { // Train every 4 steps for efficiency
          this.trainDqnModel();
        }
      }

      currentPosition = nextPosition;

      if (done) {
        episodeSuccess = (currentPosition.x === this.maze.end.x && currentPosition.y === this.maze.end.y);
        break;
      }
    }
    
    // Final position update
    this.trackAgentMovement(currentPosition);
    if (this.visualizationSettings.enabled) {
      this.agentMovedSubject.next(currentPosition);
    }

    // Update exploration rate for Q-Learning
    if (this.currentAlgorithm === AlgorithmType.QLEARNING && this.currentConfig.explorationRate > this.currentConfig.minExplorationRate) {
      this.currentConfig.explorationRate *= this.currentConfig.explorationDecay;
    }

    this.totalRewardsHistory.push(totalReward);
    this.successHistory.push(episodeSuccess);
    this.currentEpisode++;

    // Throttled stats updates for performance
    const shouldUpdateStats = this.currentEpisode % Math.max(1, Math.floor(this.currentConfig.episodes / 100)) === 0 || 
                             this.currentEpisode === this.currentConfig.episodes;
    
    if (shouldUpdateStats) {
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
    }

    // Update target model periodically for DQN
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
    
    await this.dqnModel.fit(batch.states, batch.targets, {
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
    this.isPaused = false;
    this.trainingStatusSubject.next({ isRunning: false, isPaused: false, message: message ?? 'Training stopped by user.' });
    console.log(message ?? 'AI Training Stopped');
  }
  
  getQTableArray(): {state: string, qValues: number[]}[] {
      return Object.entries(this.qTable).map(([state, qValues]) => ({state, qValues}));
  }

  async testModel(maze?: Maze, maxSteps: number = 200): Promise<void> {
    if (!this.currentAlgorithm) {
      console.error('No trained model available');
      this.testingStatusSubject.next({ isRunning: false, message: 'No trained model available' });
      return;
    }

    const testMaze = maze || this.maze;
    if (!testMaze) {
      console.error('No maze available for testing');
      this.testingStatusSubject.next({ isRunning: false, message: 'No maze available for testing' });
      return;
    }

    this.testingStatusSubject.next({ isRunning: true, message: 'Testing model...' });
    let currentPosition = { ...testMaze.start };
    let totalReward = 0;
    let steps = 0;
    let path: Position[] = [currentPosition];
    let success = false;

    try {
      while (steps < maxSteps) {
        this.agentMovedSubject.next(currentPosition);
        await new Promise(resolve => setTimeout(resolve, 100)); // Delay for visualization

        const state = this.getCurrentState(currentPosition);
        const action = this.chooseBestAction(state); // Use greedy policy for testing
        const { nextPosition, reward, done } = this.takeAction(currentPosition, action);
        
        totalReward += reward;
        currentPosition = nextPosition;
        path.push(currentPosition);
        steps++;

        if (done) {
          success = (currentPosition.x === testMaze.end.x && currentPosition.y === testMaze.end.y);
          break;
        }
      }

      this.testStatsSubject.next({
        totalSteps: steps,
        success,
        reward: totalReward,
        path
      });

      this.testingStatusSubject.next({ 
        isRunning: false, 
        message: success ? 'Test completed successfully!' : 'Test completed - goal not reached.'
      });
    } catch (error) {
      console.error('Error during testing:', error);
      this.testingStatusSubject.next({ isRunning: false, message: 'Error during testing.' });
    }
  }

  private chooseBestAction(state: State): Action {
    if (this.currentAlgorithm === AlgorithmType.QLEARNING) {
      const stateKey = this.getStateKey(state);
      const qValues = this.qTable[stateKey] || [0, 0, 0, 0];
      return qValues.indexOf(Math.max(...qValues)) as Action;
    } else if (this.currentAlgorithm === AlgorithmType.DQN && this.dqnModel) {
      return tf.tidy(() => {
        const stateInput = tf.tensor2d([this.getStateRepresentation(state.position)]);
        const prediction = this.dqnModel!.predict(stateInput) as tf.Tensor;
        const action = prediction.argMax(1).dataSync()[0];
        return action as Action;
      });
    }
    return Action.UP; // Default fallback
  }

  // Optional: Method to get the current training progress
  getCurrentProgress(): number {
    if (!this.currentConfig) return 0;
    return (this.currentEpisode / this.currentConfig.episodes) * 100;
  }

  // Optional: Method to save the trained model
  async saveModel(): Promise<void> {
    if (this.currentAlgorithm === AlgorithmType.DQN && this.dqnModel) {
      try {
        await this.dqnModel.save('localstorage://maze-solver-dqn');
        console.log('DQN model saved successfully');
      } catch (error) {
        console.error('Error saving DQN model:', error);
      }
    } else if (this.currentAlgorithm === AlgorithmType.QLEARNING) {
      try {
        localStorage.setItem('maze-solver-qtable', JSON.stringify(this.qTable));
        console.log('Q-table saved successfully');
      } catch (error) {
        console.error('Error saving Q-table:', error);
      }
    }
  }

  // Optional: Method to load a saved model
  async loadModel(): Promise<void> {
    if (this.currentAlgorithm === AlgorithmType.DQN) {
      try {
        const loadedModel = await tf.loadLayersModel('localstorage://maze-solver-dqn');
        if (loadedModel instanceof tf.Sequential) {
          this.dqnModel = loadedModel;
          
          // Safely get input and output shapes
          const inputShape = loadedModel.inputs[0].shape;
          const outputShape = loadedModel.outputs[0].shape;
          
          if (!inputShape || !outputShape) {
            throw new Error('Invalid model shapes');
          }

          // Create a new target model with the same architecture
          this.targetDqnModel = this.createDqnModel(
            inputShape[1] as number, // Get the feature dimension
            outputShape[1] as number, // Get the number of actions
            this.currentNnConfig || {
              hiddenLayers: [128, 128],
              activation: 'relu' as ActivationType,
              optimizer: 'adam',
              learningRate: 0.001,
              batchSize: 32,
              memorySize: 10000,
              targetUpdateFrequency: 10
            }
          );
          
          // Copy weights from loaded model to target model
          this.updateTargetModel();
          console.log('DQN model loaded successfully');
        } else {
          throw new Error('Loaded model is not a Sequential model');
        }
      } catch (error) {
        console.error('Error loading DQN model:', error);
      }
    } else if (this.currentAlgorithm === AlgorithmType.QLEARNING) {
      try {
        const savedQTable = localStorage.getItem('maze-solver-qtable');
        if (savedQTable) {
          this.qTable = JSON.parse(savedQTable);
          console.log('Q-table loaded successfully');
        }
      } catch (error) {
        console.error('Error loading Q-table:', error);
      }
    }
  }

  // Enhanced visualization methods
  private trackAgentMovement(position: Position): void {
    if (!this.visualizationSettings.enabled) return;
    
    const key = `${position.x},${position.y}`;
    this.visitedCells.set(key, (this.visitedCells.get(key) ?? 0) + 1);
    
    // Update exploration heatmap
    if (this.visualizationSettings.showExplorationHeatmap) {
      this.explorationHeatmap.set(key, Date.now());
    }
  }

  private shouldUpdateVisualization(): boolean {
    if (!this.visualizationSettings.enabled) return false;
    
    const now = Date.now();
    const timeSinceLastUpdate = now - this.lastVisualizationUpdate;
    const requiredInterval = this.getVisualizationInterval();
    
    return timeSinceLastUpdate >= requiredInterval;
  }

  private getVisualizationInterval(): number {
    // Calculate update interval based on speed and performance mode
    const baseInterval = this.visualizationSettings.performanceMode ? 200 : 50;
    const speedMultiplier = (100 - this.visualizationSettings.speed) / 100;
    return Math.max(16, baseInterval * speedMultiplier); // Minimum 16ms (60fps)
  }

  private clearVisualizationHistory(): void {
    this.visitedCells.clear();
    this.explorationHeatmap.clear();
    this.visualizationHistory = [];
  }

  setVisualizationSettings(settings: VisualizationSettings): void {
    this.visualizationSettings = { ...this.visualizationSettings, ...settings };
    
    // Clear history if switching modes
    if (!settings.enabled) {
      this.clearVisualizationHistory();
    }
    
    console.log('Visualization settings updated:', this.visualizationSettings);
  }

  setVisualizationSpeed(speed: number): void {
    this.visualizationSpeed = Math.max(1, Math.min(100, speed));
    this.visualizationSettings.speed = this.visualizationSpeed;
    console.log('Visualization speed set to:', this.visualizationSpeed);
  }

  setOptimizedVisualization(enabled: boolean): void {
    this.enableOptimizedVisualization = enabled;
    this.visualizationSettings.performanceMode = !enabled;
    console.log('Optimized visualization:', enabled ? 'enabled' : 'disabled');
  }

  getVisualizationData(): any {
    return {
      visitedCells: Array.from(this.visitedCells.entries()),
      explorationHeatmap: Array.from(this.explorationHeatmap.entries()),
      settings: this.visualizationSettings
    };
  }
}