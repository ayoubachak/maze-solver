import { Injectable } from '@angular/core';
import { AlgorithmType } from '../../models/maze.model';
import { BaseAlgorithm } from './base-algorithm';
import { QLearningAlgorithm } from './qlearning-algorithm';
import { DQNAlgorithm } from './dqn-algorithm';
import { NEATAlgorithm } from './neat-algorithm';

@Injectable({
  providedIn: 'root'
})
export class AlgorithmFactory {
  private readonly algorithms = new Map<AlgorithmType, BaseAlgorithm>();

  constructor(
    private readonly qLearningAlgorithm: QLearningAlgorithm,
    private readonly dqnAlgorithm: DQNAlgorithm,
    private readonly neatAlgorithm: NEATAlgorithm
  ) {
    this.algorithms.set(AlgorithmType.QLEARNING, this.qLearningAlgorithm);
    this.algorithms.set(AlgorithmType.DQN, this.dqnAlgorithm);
    this.algorithms.set(AlgorithmType.NEAT, this.neatAlgorithm);
  }

  getAlgorithm(type: AlgorithmType): BaseAlgorithm {
    const algorithm = this.algorithms.get(type);
    if (!algorithm) {
      throw new Error(`Algorithm ${type} not found`);
    }
    return algorithm;
  }

  getAllAlgorithms(): { value: AlgorithmType; name: string; description: string; algorithm: BaseAlgorithm }[] {
    return [
      {
        value: AlgorithmType.QLEARNING,
        name: 'Q-Learning',
        description: 'A model-free reinforcement learning algorithm to learn the quality of actions.',
        algorithm: this.qLearningAlgorithm
      },
      {
        value: AlgorithmType.DQN,
        name: 'Deep Q-Network (DQN)',
        description: 'Uses a neural network to approximate Q-values, suitable for complex states.',
        algorithm: this.dqnAlgorithm
      },
      {
        value: AlgorithmType.NEAT,
        name: 'NEAT',
        description: 'NeuroEvolution of Augmenting Topologies - evolves neural network structure and weights through genetic algorithms.',
        algorithm: this.neatAlgorithm
      }
    ];
  }
}