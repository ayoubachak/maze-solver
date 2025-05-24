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
  VisualizationSettings,
  NEATConfig,
  NEATGenome,
  NEATNode,
  NEATConnection,
  NEATAgent,
  NEATSpecies,
  NEATStats,
  NEATInnovation
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

  // NEAT algorithm properties
  private neatPopulation: NEATGenome[] = [];
  private neatSpecies: NEATSpecies[] = [];
  private neatInnovationHistory: NEATInnovation[] = [];
  private neatGeneration = 0;
  private neatGlobalInnovationNumber = 0;
  private neatNodeId = 0;
  private neatBestFitness = -Infinity;
  private neatStagnationCounter = 0;
  private readonly NEAT_POPULATION_SIZE = 150;
  private readonly NEAT_MAX_STAGNATION = 15;
  private readonly NEAT_SPECIES_THRESHOLD = 3.0;
  private readonly NEAT_SURVIVAL_RATE = 0.2;
  private readonly NEAT_MUTATION_RATE = 0.8;
  private readonly NEAT_CROSSOVER_RATE = 0.75;
  private readonly NEAT_MAX_STEPS_PER_AGENT = 200;

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

  // NEAT Algorithm Properties - consolidated
  private neatConfig: NEATConfig | null = null;
  private neatAgents: NEATAgent[] = [];
  private neatInnovationNumber = 0;
  private readonly neatStatsSubject = new BehaviorSubject<NEATStats | null>(null);
  neatStats$ = this.neatStatsSubject.asObservable();

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
    config: TrainingConfig | NEATConfig, 
    nnConfig?: NeuralNetworkConfig
  ): void {
    if (!this.maze) {
      console.error('Maze not initialized for AI training');
      return;
    }
    this.stopTraining(); // Stop any ongoing training

    // Reset all stats and visualization when switching algorithms
    this.resetTrainingState();

    this.currentAlgorithm = algorithm;
    
    if (algorithm === AlgorithmType.NEAT) {
      this.neatConfig = config as NEATConfig;
      this.currentConfig = null; // NEAT doesn't use TrainingConfig
    } else {
      this.currentConfig = config as TrainingConfig;
      this.neatConfig = null;
      this.currentNnConfig = nnConfig as NeuralNetworkConfigInternal;
    }
    this.currentEpisode = 0;
    this.totalRewardsHistory = [];
    this.successHistory = [];
    this.isPaused = false;

    // Reset visualization state
    this.clearVisualizationHistory();
    this.resetAgentPosition();

    this.trainingStatusSubject.next({ isRunning: true, isPaused: false });

    if (algorithm === AlgorithmType.DQN) {
      this.initializeDqnModel();
    } else if (algorithm === AlgorithmType.NEAT) {
      this.initializeNEAT();
    }

    // Run training outside Angular zone to prevent excessive change detection
    this.ngZone.runOutsideAngular(() => {
      if (algorithm === AlgorithmType.NEAT) {
        // For NEAT, run generations sequentially to handle async evaluation
        this.runNEATTrainingLoop();
      } else {
        this.trainingSubscription = interval(10) // Small delay for visualization updates
          .subscribe(() => {
            if (!this.isPaused && this.currentEpisode < this.currentConfig!.episodes) {
              this.runEpisode();
            } else if (!this.isPaused && this.currentEpisode >= this.currentConfig!.episodes) {
              this.stopTraining('Training completed: All episodes finished.');
            }
          });
      }
    });
  }

  private resetTrainingState(): void {
    // Clear previous training stats
    this.trainingStatsSubject.next(null);
    this.testStatsSubject.next(null);
    
    // Clear algorithm-specific state
    this.qTable = {};
    this.dqnModel = null;
    this.targetDqnModel = null;
    this.replayMemory = [];
    
    // Reset NEAT state
    this.neatPopulation = [];
    this.neatSpecies = [];
    this.neatInnovationHistory = [];
    this.neatGeneration = 0;
    this.neatGlobalInnovationNumber = 0;
    this.neatNodeId = 0;
    this.neatBestFitness = -Infinity;
    this.neatStagnationCounter = 0;
    this.neatStatsSubject.next(null);
    
    // Reset training progress
    this.currentEpisode = 0;
    this.totalRewardsHistory = [];
    this.successHistory = [];
  }

  private resetAgentPosition(): void {
    if (this.maze) {
      // Reset agent to start position and clear any position markers
      this.agentMovedSubject.next(this.maze.start);
    }
  }

  private async runNEATTrainingLoop(): Promise<void> {
    const maxGenerations = this.neatConfig?.populationSize || 100; // Use a sensible default for generations
    const runGeneration = async () => {
      if (!this.isPaused && this.neatConfig && this.neatGeneration < maxGenerations) {
        console.log(`Running NEAT generation ${this.neatGeneration + 1}`);
        await this.runNEATGeneration();
        
        // Schedule next generation
        setTimeout(() => runGeneration(), 100);
      } else if (!this.isPaused && this.neatConfig && this.neatGeneration >= maxGenerations) {
        this.stopTraining('NEAT training completed: All generations finished.');
      }
    };

    await runGeneration();
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

    // Clear visited cells tracking for each episode to improve exploration reward
    this.visitedCells.clear();

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
        if (this.replayMemory.length >= this.currentNnConfig!.batchSize && steps % 4 === 0) {
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

    // Enhanced exploration rate decay for DQN
    if (this.currentAlgorithm === AlgorithmType.DQN) {
      // Use exponential decay with minimum exploration rate
      const decayRate = 0.995;
      const minExploration = 0.05;
      this.currentConfig.explorationRate = Math.max(minExploration, this.currentConfig.explorationRate * decayRate);
    } else if (this.currentAlgorithm === AlgorithmType.QLEARNING && this.currentConfig.explorationRate > this.currentConfig.minExplorationRate) {
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
    const originalDistance = Math.abs(currentPosition.x - this.maze.end.x) + Math.abs(currentPosition.y - this.maze.end.y);

    switch (action) {
      case Action.UP:    y--; break;
      case Action.DOWN:  y++; break;
      case Action.LEFT:  x--; break;
      case Action.RIGHT: x++; break;
    }

    let reward = -0.04; // Smaller step penalty to encourage exploration
    let done = false;

    if (this.isWall(x, y)) {
      reward = -0.5; // Reduced wall penalty 
      x = currentPosition.x; // Stay in place
      y = currentPosition.y;
    } else {
      // Calculate new distance to goal
      const newDistance = Math.abs(x - this.maze.end.x) + Math.abs(y - this.maze.end.y);
      
      if (x === this.maze.end.x && y === this.maze.end.y) {
        reward = 10.0; // Large reward for reaching goal
        done = true;
      } else {
        // Reward for getting closer to goal, penalty for moving away
        const distanceReward = (originalDistance - newDistance) * 0.1;
        reward += distanceReward;
        
        // Additional small reward for exploration (visiting new cells)
        const cellKey = `${x},${y}`;
        if (!this.visitedCells.has(cellKey)) {
          reward += 0.02; // Small exploration bonus
        }
        
        // Penalty for staying in the same place too long
        const visitCount = this.visitedCells.get(cellKey) ?? 0;
        if (visitCount > 3) {
          reward -= 0.01 * visitCount; // Increasing penalty for revisiting
        }
      }
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
    
    const stateArray: number[] = [];
    
    // 1. Current position (normalized)
    stateArray.push(position.x / this.maze.width);
    stateArray.push(position.y / this.maze.height);
    
    // 2. Distance to goal (normalized)
    const distanceX = (this.maze.end.x - position.x) / this.maze.width;
    const distanceY = (this.maze.end.y - position.y) / this.maze.height;
    stateArray.push(distanceX);
    stateArray.push(distanceY);
    
    // 3. Manhattan distance to goal (normalized)
    const manhattanDistance = (Math.abs(this.maze.end.x - position.x) + Math.abs(this.maze.end.y - position.y)) / (this.maze.width + this.maze.height);
    stateArray.push(manhattanDistance);
    
    // 4. Immediate surroundings (3x3 grid around agent)
    for (let dy = -1; dy <= 1; dy++) {
      for (let dx = -1; dx <= 1; dx++) {
        if (dx === 0 && dy === 0) continue; // Skip current position
        const x = position.x + dx;
        const y = position.y + dy;
        stateArray.push(this.isWall(x, y) ? 1 : 0);
      }
    }
    
    // 5. Direction to goal (unit vector)
    const goalDist = Math.sqrt(distanceX * distanceX + distanceY * distanceY);
    if (goalDist > 0) {
      stateArray.push(distanceX / goalDist); // Normalized direction X
      stateArray.push(distanceY / goalDist); // Normalized direction Y
    } else {
      stateArray.push(0);
      stateArray.push(0);
    }
    
    // 6. Visibility in four directions (how far can we see)
    const directions = [
      { dx: 0, dy: -1 }, // Up
      { dx: 0, dy: 1 },  // Down
      { dx: -1, dy: 0 }, // Left
      { dx: 1, dy: 0 }   // Right
    ];
    
    directions.forEach(dir => {
      let distance = 0;
      let x = position.x + dir.dx;
      let y = position.y + dir.dy;
      
      while (!this.isWall(x, y) && distance < 10) {
        distance++;
        x += dir.dx;
        y += dir.dy;
      }
      stateArray.push(distance / 10); // Normalized visibility distance
    });
    
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
    const key = `${position.x},${position.y}`;
    this.visitedCells.set(key, (this.visitedCells.get(key) || 0) + 1);
    this.explorationHeatmap.set(key, Math.min(1, (this.explorationHeatmap.get(key) || 0) + 0.1));
    
    this.visualizationHistory.push({ position, timestamp: Date.now() });
    if (this.visualizationHistory.length > this.visualizationSettings.maxHistorySize) {
      this.visualizationHistory.shift();
    }
  }

  private shouldUpdateVisualization(): boolean {
    if (!this.visualizationSettings.enabled) return false;
    const now = Date.now();
    const interval = this.getVisualizationInterval();
    return now - this.lastVisualizationUpdate >= interval;
  }

  private getVisualizationInterval(): number {
    const baseInterval = 100;
    const speedMultiplier = this.visualizationSettings.speed / 100;
    return Math.max(10, baseInterval * (1 - speedMultiplier));
  }

  private clearVisualizationHistory(): void {
    this.visitedCells.clear();
    this.explorationHeatmap.clear();
    this.visualizationHistory = [];
  }

  setVisualizationSettings(settings: VisualizationSettings): void {
    this.visualizationSettings = { ...this.visualizationSettings, ...settings };
  }

  setVisualizationSpeed(speed: number): void {
    this.visualizationSettings.speed = Math.max(1, Math.min(100, speed));
  }

  setOptimizedVisualization(enabled: boolean): void {
    this.enableOptimizedVisualization = enabled;
  }

  getVisualizationData(): any {
    return {
      visitedCells: Array.from(this.visitedCells.entries()),
      explorationHeatmap: Array.from(this.explorationHeatmap.entries()),
      agentPath: this.visualizationHistory.map(h => h.position),
      performance: {
        fps: 1000 / Math.max(1, Date.now() - this.lastVisualizationUpdateTime),
        updateCount: this.frameSkipCount,
        optimized: this.enableOptimizedVisualization
      }
    };
  }

  // NEAT Algorithm Implementation
  private initializeNEAT(): void {
    console.log('Initializing NEAT algorithm...');
    this.initializeNEATPopulation();
    console.log(`NEAT initialized with ${this.neatPopulation.length} genomes in ${this.neatSpecies.length} species`);
  }

  private initializeNEATPopulation(): void {
    this.neatPopulation = [];
    this.neatSpecies = [];
    this.neatInnovationHistory = [];
    this.neatGeneration = 0;
    this.neatGlobalInnovationNumber = 0;
    this.neatNodeId = 0;
    this.neatBestFitness = -Infinity;
    this.neatStagnationCounter = 0;

    // Create initial population with minimal structure
    for (let i = 0; i < this.NEAT_POPULATION_SIZE; i++) {
      const genome = this.createMinimalGenome(i);
      this.neatPopulation.push(genome);
    }

    // Initial speciation
    this.speciatePopulation();
  }

  private createMinimalGenome(id: number): NEATGenome {
    const nodes: NEATNode[] = [];
    const connections: NEATConnection[] = [];

    // Input nodes (4: up, down, left, right wall detection)
    for (let i = 0; i < 4; i++) {
      nodes.push({
        id: this.neatNodeId++,
        type: 'input',
        x: 0,
        y: i / 3,
        value: 0,
        bias: 0
      });
    }

    // Output nodes (4: up, down, left, right actions)
    for (let i = 0; i < 4; i++) {
      nodes.push({
        id: this.neatNodeId++,
        type: 'output',
        x: 1,
        y: i / 3,
        value: 0,
        bias: Math.random() * 2 - 1
      });
    }

    // Create initial connections between inputs and outputs
    for (let input = 0; input < 4; input++) {
      for (let output = 4; output < 8; output++) {
        connections.push({
          innovationNumber: this.getInnovationNumber(input, output),
          inputNode: input,
          outputNode: output,
          weight: Math.random() * 4 - 2,
          enabled: true
        });
      }
    }

    return {
      id,
      nodes,
      connections,
      fitness: 0,
      adjustedFitness: 0,
      species: -1,
      generation: this.neatGeneration
    };
  }

  private getInnovationNumber(inputNode: number, outputNode: number, newNodeId?: number): number {
    // Check if this innovation already exists
    const existing = this.neatInnovationHistory.find(
      inn => inn.inputNode === inputNode && inn.outputNode === outputNode && inn.newNodeId === newNodeId
    );

    if (existing) {
      return existing.innovationNumber;
    }

    // Create new innovation
    const innovation: NEATInnovation = {
      innovationNumber: this.neatGlobalInnovationNumber++,
      inputNode,
      outputNode,
      newNodeId
    };

    this.neatInnovationHistory.push(innovation);
    return innovation.innovationNumber;
  }

  private speciatePopulation(): void {
    this.neatSpecies = [];

    for (const genome of this.neatPopulation) {
      let foundSpecies = false;

      for (const species of this.neatSpecies) {
        if (this.calculateGenomeDistance(genome, species.representative) < this.NEAT_SPECIES_THRESHOLD) {
          species.members.push(genome);
          genome.species = species.id;
          foundSpecies = true;
          break;
        }
      }

      if (!foundSpecies) {
        const newSpecies: NEATSpecies = {
          id: this.neatSpecies.length,
          representative: genome,
          members: [genome],
          averageFitness: 0,
          staleness: 0,
          topFitness: 0
        };
        this.neatSpecies.push(newSpecies);
        genome.species = newSpecies.id;
      }
    }
  }

  private calculateGenomeDistance(genome1: NEATGenome, genome2: NEATGenome): number {
    const c1 = 1.0; // Excess coefficient
    const c2 = 1.0; // Disjoint coefficient
    const c3 = 0.4; // Weight coefficient

    const connections1 = new Map(genome1.connections.map(c => [c.innovationNumber, c]));
    const connections2 = new Map(genome2.connections.map(c => [c.innovationNumber, c]));

    const allInnovations = new Set([...connections1.keys(), ...connections2.keys()]);
    const maxInnovation = Math.max(...allInnovations);

    let excess = 0;
    let disjoint = 0;
    let weightDiff = 0;
    let matching = 0;

    for (const innovation of allInnovations) {
      const conn1 = connections1.get(innovation);
      const conn2 = connections2.get(innovation);

      if (conn1 && conn2) {
        // Matching connection
        weightDiff += Math.abs(conn1.weight - conn2.weight);
        matching++;
      } else if (innovation > maxInnovation - 10) {
        // Excess connection
        excess++;
      } else {
        // Disjoint connection
        disjoint++;
      }
    }

    const N = Math.max(genome1.connections.length, genome2.connections.length, 1);
    const avgWeightDiff = matching > 0 ? weightDiff / matching : 0;

    return (c1 * excess / N) + (c2 * disjoint / N) + (c3 * avgWeightDiff);
  }

  private evaluateNEATGenome(genome: NEATGenome): Promise<number> {
    return new Promise((resolve) => {
      if (!this.maze) {
        resolve(0);
        return;
      }

      let currentPosition = { ...this.maze.start };
      let fitness = 0;
      let steps = 0;
      const maxSteps = this.NEAT_MAX_STEPS_PER_AGENT;
      const visitedPositions = new Set<string>();

      const evaluateStep = () => {
        if (steps >= maxSteps) {
          resolve(fitness);
          return;
        }

        const state = this.getCurrentState(currentPosition);
        const action = this.activateNEATGenome(genome, state);
        const { nextPosition, reward, done } = this.takeAction(currentPosition, action);

        currentPosition = nextPosition;
        fitness += reward;
        steps++;

        // Encourage exploration
        const posKey = `${currentPosition.x},${currentPosition.y}`;
        if (!visitedPositions.has(posKey)) {
          visitedPositions.add(posKey);
          fitness += 1; // Exploration bonus
        }

        // Notify visualization
        this.agentMovedSubject.next(currentPosition);

        if (done) {
          fitness += 100; // Goal reached bonus
          resolve(fitness);
          return;
        }

        // Continue evaluation asynchronously
        setTimeout(evaluateStep, 1);
      };

      evaluateStep();
    });
  }

  private activateNEATGenome(genome: NEATGenome, state: State): Action {
    // Reset node values
    genome.nodes.forEach(node => {
      if (node.type !== 'input') {
        node.value = 0;
      }
    });

    // Set input values
    genome.nodes[0].value = state.walls[0] ? 1 : 0; // Up
    genome.nodes[1].value = state.walls[1] ? 1 : 0; // Down
    genome.nodes[2].value = state.walls[2] ? 1 : 0; // Left
    genome.nodes[3].value = state.walls[3] ? 1 : 0; // Right

    // Activate network
    const sortedNodes = [...genome.nodes].sort((a, b) => {
      if (a.type === 'input' && b.type !== 'input') return -1;
      if (a.type !== 'input' && b.type === 'input') return 1;
      if (a.type === 'output' && b.type !== 'output') return 1;
      if (a.type !== 'output' && b.type === 'output') return -1;
      return 0;
    });

    for (const node of sortedNodes) {
      if (node.type === 'input') continue;

      let sum = node.bias;
      for (const connection of genome.connections) {
        if (connection.outputNode === node.id && connection.enabled) {
          const inputNode = genome.nodes.find(n => n.id === connection.inputNode);
          if (inputNode) {
            sum += inputNode.value * connection.weight;
          }
        }
      }

      node.value = this.sigmoid(sum);
    }

    // Find best output
    const outputNodes = genome.nodes.filter(n => n.type === 'output');
    let bestAction = 0;
    let bestValue = outputNodes[0].value;

    for (let i = 1; i < outputNodes.length; i++) {
      if (outputNodes[i].value > bestValue) {
        bestValue = outputNodes[i].value;
        bestAction = i;
      }
    }

    return bestAction as Action;
  }

  private sigmoid(x: number): number {
    return 1 / (1 + Math.exp(-x));
  }

  private async runNEATGeneration(): Promise<void> {
    // Evaluate all genomes
    const evaluationPromises = this.neatPopulation.map(genome => 
      this.evaluateNEATGenome(genome)
    );

    const fitnesses = await Promise.all(evaluationPromises);
    
    // Assign fitnesses
    this.neatPopulation.forEach((genome, index) => {
      genome.fitness = fitnesses[index];
    });

    // Update best fitness
    const currentBest = Math.max(...fitnesses);
    if (currentBest > this.neatBestFitness) {
      this.neatBestFitness = currentBest;
      this.neatStagnationCounter = 0;
    } else {
      this.neatStagnationCounter++;
    }

    // Calculate species fitness
    this.calculateSpeciesFitness();

    // Create next generation
    this.reproduceSpecies();

    // Mutate population
    this.mutatePopulation();

    // Re-speciate
    this.speciatePopulation();

    this.neatGeneration++;

    // Update stats
    const stats = this.getNEATStats();
    this.neatStatsSubject.next(stats);
    this.trainingStatsSubject.next({
      episode: stats.generation,
      totalReward: stats.bestFitness,
      steps: 0,
      explorationRate: 0,
      success: stats.bestFitness > 50,
      averageReward: stats.averageFitness,
      successRate: (this.neatPopulation.filter(g => g.fitness > 50).length / this.neatPopulation.length) * 100
    });
  }

  private calculateSpeciesFitness(): void {
    for (const species of this.neatSpecies) {
      const totalFitness = species.members.reduce((sum, genome) => sum + genome.fitness, 0);
      species.averageFitness = totalFitness / species.members.length;
      
      const maxFitness = Math.max(...species.members.map(g => g.fitness));
      if (maxFitness > species.topFitness) {
        species.topFitness = maxFitness;
        species.staleness = 0;
      } else {
        species.staleness++;
      }
    }
  }

  private reproduceSpecies(): void {
    const newPopulation: NEATGenome[] = [];
    let nextId = 0;

    // Calculate total adjusted fitness
    const totalAdjustedFitness = this.neatSpecies.reduce((sum, species) => {
      return sum + species.averageFitness * species.members.length;
    }, 0);

    for (const species of this.neatSpecies) {
      if (species.staleness > this.NEAT_MAX_STAGNATION) continue;

      const speciesSize = Math.floor(
        (species.averageFitness * species.members.length / totalAdjustedFitness) * this.NEAT_POPULATION_SIZE
      );

      if (speciesSize === 0) continue;

      // Sort by fitness
      species.members.sort((a, b) => b.fitness - a.fitness);

      // Keep champion
      if (species.members.length > 5) {
        newPopulation.push({ ...species.members[0], id: nextId++, generation: this.neatGeneration });
      }

      // Generate offspring
      for (let i = newPopulation.length; i < speciesSize && newPopulation.length < this.NEAT_POPULATION_SIZE; i++) {
        let offspring: NEATGenome;

        if (Math.random() < this.NEAT_CROSSOVER_RATE && species.members.length > 1) {
          const parent1 = this.selectParent(species.members);
          const parent2 = this.selectParent(species.members);
          offspring = this.crossover(parent1, parent2, nextId++);
        } else {
          const parent = this.selectParent(species.members);
          offspring = { ...parent, id: nextId++, generation: this.neatGeneration };
        }

        newPopulation.push(offspring);
      }
    }

    // Fill remaining slots with random genomes
    while (newPopulation.length < this.NEAT_POPULATION_SIZE) {
      newPopulation.push(this.createMinimalGenome(nextId++));
    }

    this.neatPopulation = newPopulation;
  }

  private selectParent(species: NEATGenome[]): NEATGenome {
    const survivalCount = Math.max(1, Math.floor(species.length * this.NEAT_SURVIVAL_RATE));
    const randomIndex = Math.floor(Math.random() * survivalCount);
    return species[randomIndex];
  }

  private crossover(parent1: NEATGenome, parent2: NEATGenome, id: number): NEATGenome {
    const offspring: NEATGenome = {
      id,
      nodes: [],
      connections: [],
      fitness: 0,
      adjustedFitness: 0,
      species: -1,
      generation: this.neatGeneration
    };

    // Copy nodes from more fit parent
    const fitterParent = parent1.fitness >= parent2.fitness ? parent1 : parent2;
    offspring.nodes = fitterParent.nodes.map(node => ({ ...node }));

    // Crossover connections
    const connections1 = new Map(parent1.connections.map(c => [c.innovationNumber, c]));
    const connections2 = new Map(parent2.connections.map(c => [c.innovationNumber, c]));
    const allInnovations = new Set([...connections1.keys(), ...connections2.keys()]);

    for (const innovation of allInnovations) {
      const conn1 = connections1.get(innovation);
      const conn2 = connections2.get(innovation);

      if (conn1 && conn2) {
        // Matching connection - randomly choose parent
        const chosen = Math.random() < 0.5 ? conn1 : conn2;
        offspring.connections.push({ ...chosen });
      } else if (conn1 && parent1.fitness >= parent2.fitness) {
        // Excess/disjoint from fitter parent
        offspring.connections.push({ ...conn1 });
      } else if (conn2 && parent2.fitness >= parent1.fitness) {
        // Excess/disjoint from fitter parent
        offspring.connections.push({ ...conn2 });
      }
    }

    return offspring;
  }

  private mutatePopulation(): void {
    for (const genome of this.neatPopulation) {
      if (Math.random() < this.NEAT_MUTATION_RATE) {
        this.mutateGenome(genome);
      }
    }
  }

  private mutateGenome(genome: NEATGenome): void {
    const mutations = Math.random();

    if (mutations < 0.05) {
      this.addNodeMutation(genome);
    } else if (mutations < 0.15) {
      this.addConnectionMutation(genome);
    } else {
      this.weightMutation(genome);
    }
  }

  private addNodeMutation(genome: NEATGenome): void {
    if (genome.connections.length === 0) return;

    const connection = genome.connections[Math.floor(Math.random() * genome.connections.length)];
    if (!connection.enabled) return;

    // Disable old connection
    connection.enabled = false;

    // Create new node
    const newNode: NEATNode = {
      id: this.neatNodeId++,
      type: 'hidden',
      x: Math.random(),
      y: Math.random(),
      value: 0,
      bias: 0
    };
    genome.nodes.push(newNode);

    // Create new connections
    genome.connections.push({
      innovationNumber: this.getInnovationNumber(connection.inputNode, newNode.id),
      inputNode: connection.inputNode,
      outputNode: newNode.id,
      weight: 1.0,
      enabled: true
    });

    genome.connections.push({
      innovationNumber: this.getInnovationNumber(newNode.id, connection.outputNode),
      inputNode: newNode.id,
      outputNode: connection.outputNode,
      weight: connection.weight,
      enabled: true
    });
  }

  private addConnectionMutation(genome: NEATGenome): void {
    const inputNodes = genome.nodes.filter(n => n.type !== 'output');
    const outputNodes = genome.nodes.filter(n => n.type !== 'input');

    if (inputNodes.length === 0 || outputNodes.length === 0) return;

    const inputNode = inputNodes[Math.floor(Math.random() * inputNodes.length)];
    const outputNode = outputNodes[Math.floor(Math.random() * outputNodes.length)];

    // Check if connection already exists
    const exists = genome.connections.some(
      c => c.inputNode === inputNode.id && c.outputNode === outputNode.id
    );

    if (!exists) {
      genome.connections.push({
        innovationNumber: this.getInnovationNumber(inputNode.id, outputNode.id),
        inputNode: inputNode.id,
        outputNode: outputNode.id,
        weight: Math.random() * 4 - 2,
        enabled: true
      });
    }
  }

  private weightMutation(genome: NEATGenome): void {
    for (const connection of genome.connections) {
      if (Math.random() < 0.9) {
        // Small perturbation
        connection.weight += (Math.random() * 2 - 1) * 0.1;
      } else {
        // Large random change
        connection.weight = Math.random() * 4 - 2;
      }
    }
  }

  private getNEATStats(): NEATStats {
    const fitnesses = this.neatPopulation.map(g => g.fitness);
    const bestGenome = this.neatPopulation.reduce((best, current) => 
      current.fitness > best.fitness ? current : best
    );

    return {
      generation: this.neatGeneration,
      bestFitness: Math.max(...fitnesses),
      averageFitness: fitnesses.reduce((sum, f) => sum + f, 0) / fitnesses.length,
      speciesCount: this.neatSpecies.length,
      populationSize: this.neatPopulation.length,
      stagnationCounter: this.neatStagnationCounter,
      topAgent: {
        id: bestGenome.id,
        fitness: bestGenome.fitness,
        steps: 0,
        success: bestGenome.fitness > 50
      }
    };
  }

  // Public method to get NEAT visualization data
  getNEATVisualizationData(): any {
    if (this.neatPopulation.length === 0) return null;

    const bestGenome = this.neatPopulation.reduce((best, current) => 
      current.fitness > best.fitness ? current : best
    );

    return {
      bestGenome,
      species: this.neatSpecies.map(s => ({
        id: s.id,
        size: s.members.length,
        averageFitness: s.averageFitness,
        staleness: s.staleness
      })),
      generation: this.neatGeneration,
      populationStats: this.getNEATStats()
    };
  }
}