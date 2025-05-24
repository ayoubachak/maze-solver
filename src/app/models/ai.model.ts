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

export interface VisualizationSettings {
  enabled: boolean;
  speed: number; // 1-100
  showPath: boolean;
  showExplorationHeatmap: boolean;
  showAgentTrail: boolean;
  performanceMode: boolean;
  adaptiveQuality: boolean;
  maxHistorySize: number;
}

export interface VisualizationData {
  agentPath: Position[];
  exploredCells: Position[];
  heatmap: Map<string, number>;
  performanceMetrics: {
    fps: number;
    updateCount: number;
    memoryUsage: number;
  };
}

// NEAT (NeuroEvolution of Augmenting Topologies) interfaces
export interface NEATGenome {
  id: number;
  nodes: NEATNode[];
  connections: NEATConnection[];
  fitness: number;
  adjustedFitness: number;
  species: number;
  generation: number;
}

export interface NEATNode {
  id: number;
  type: 'input' | 'hidden' | 'output';
  x: number; // For visualization
  y: number; // For visualization
  value: number;
  bias: number;
}

export interface NEATConnection {
  innovationNumber: number;
  inputNode: number;
  outputNode: number;
  weight: number;
  enabled: boolean;
}

export interface NEATSpecies {
  id: number;
  representative: NEATGenome;
  members: NEATGenome[];
  averageFitness: number;
  staleness: number;
  topFitness: number;
}

export interface NEATInnovation {
  innovationNumber: number;
  inputNode: number;
  outputNode: number;
  newNodeId?: number;
}

export interface NEATAgent {
  id: number;
  genome: NEATGenome;
  fitness: number;
  steps: number;
  success: boolean;
}

export interface NEATConfig {
  populationSize: number;
  maxStagnation: number;
  speciesThreshold: number;
  survivalRate: number;
  mutationRate: number;
  crossoverRate: number;
  maxStepsPerAgent: number;
  excessCoefficient: number;
  disjointCoefficient: number;
  weightCoefficient: number;
  addNodeMutationRate: number;
  addConnectionMutationRate: number;
  weightMutationRate: number;
  weightPerturbationRate: number;
}

export interface NEATStats {
  generation: number;
  bestFitness: number;
  averageFitness: number;
  speciesCount: number;
  populationSize: number;
  stagnationCounter: number;
  topAgent: {
    id: number;
    fitness: number;
    steps: number;
    success: boolean;
  };
}