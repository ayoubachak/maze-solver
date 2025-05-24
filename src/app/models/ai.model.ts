import { Position } from './maze.model';

export enum Action {
  UP = 0,
  DOWN = 1,
  LEFT = 2,
  RIGHT = 3
}

export interface State {
  position: Position;
  walls: boolean[]; // 4-element array for up, down, left, right
  distanceToGoal: number;
  previousAction?: Action;
}

export interface QTableEntry {
  state: string; // serialized state
  qValues: number[]; // Q-values for each action
}

export interface TrainingConfig {
  learningRate: number;
  discountFactor: number;
  explorationRate: number;
  explorationDecay: number;
  minExplorationRate: number;
  episodes: number;
  maxStepsPerEpisode: number;
}

export interface TrainingStats {
  episode: number;
  totalReward: number;
  steps: number;
  explorationRate: number;
  success: boolean;
  averageReward: number;
  successRate: number;
}

export interface AgentAction {
  state: State;
  action: Action;
  reward: number;
  nextState: State;
  done: boolean;
}

export interface NeuralNetworkConfig {
  hiddenLayers: number[];
  activation: string;
  optimizer: string;
  learningRate: number;
  batchSize: number;
  memorySize: number;
  targetUpdateFrequency: number;
}

export interface Experience {
  state: number[];
  action: number;
  reward: number;
  nextState: number[];
  done: boolean;
}

export interface NetworkLayer {
  neurons: Neuron[];
  type: 'input' | 'hidden' | 'output';
}

export interface Neuron {
  id: string;
  value: number;
  bias: number;
  activation: number;
  weights: number[];
  connections: Connection[];
}

export interface Connection {
  from: string;
  to: string;
  weight: number;
  active: boolean;
}

export interface NetworkVisualization {
  layers: NetworkLayer[];
  connections: Connection[];
  currentInput: number[];
  currentOutput: number[];
} 