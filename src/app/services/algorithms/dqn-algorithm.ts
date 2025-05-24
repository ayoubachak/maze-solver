import { Injectable, OnDestroy } from '@angular/core';
import { BaseAlgorithm, AlgorithmConfig } from './base-algorithm';
import { AlgorithmType } from '../../models/maze.model';
import { TrainingConfig, NeuralNetworkConfig } from '../../models/ai.model';
import { AiService } from '../ai.service';
import { Subscription } from 'rxjs';

@Injectable({
  providedIn: 'root'
})
export class DQNAlgorithm extends BaseAlgorithm implements OnDestroy {
  private trainingConfig: TrainingConfig = {
    learningRate: 0.1,
    discountFactor: 0.95,
    explorationRate: 1.0,
    explorationDecay: 0.995,
    minExplorationRate: 0.01,
    episodes: 2000,
    maxStepsPerEpisode: 300
  };

  private networkConfig: NeuralNetworkConfig = {
    hiddenLayers: [256, 256, 128],
    activation: 'relu',
    optimizer: 'adam',
    learningRate: 0.0005,
    batchSize: 64,
    memorySize: 50000,
    targetUpdateFrequency: 100
  };

  private _networkStatusMessage = 'Network visualization will appear when DQN training starts.';
  private _isDqnActive = false;
  private readonly statsSubscription?: Subscription;

  constructor(private aiService: AiService) {
    super();
    // Subscribe to AI service training stats
    this.statsSubscription = this.aiService.trainingStats$.subscribe(stats => {
      this.trainingStatsSubject.next(stats);
    });
  }

  get algorithmType(): AlgorithmType {
    return AlgorithmType.DQN;
  }

  get qTableSize(): string {
    return 'N/A (Neural Network)';
  }

  get bestActionConfidence(): string {
    return 'N/A (Neural Network)';
  }

  get networkStatusMessage(): string {
    return this._networkStatusMessage;
  }

  get isDqnActive(): boolean {
    return this._isDqnActive;
  }

  getDefaultTrainingConfig(): TrainingConfig {
    return { ...this.trainingConfig };
  }

  getDefaultNetworkConfig(): NeuralNetworkConfig {
    return { ...this.networkConfig };
  }

  initializeConfiguration(algorithmConfig: AlgorithmConfig): void {
    if (algorithmConfig.trainingConfig) {
      this.trainingConfig = { ...algorithmConfig.trainingConfig };
    }
    if (algorithmConfig.neuralNetworkConfig) {
      this.networkConfig = { ...algorithmConfig.neuralNetworkConfig };
    }
  }

  startTraining(): void {
    this._isDqnActive = true;
    this._networkStatusMessage = 'Neural network is active and learning...';
    // Pass both trainingConfig and networkConfig to respect episode limits
    this.aiService.startTraining(AlgorithmType.DQN, this.trainingConfig, this.networkConfig);
  }

  pauseTraining(): void {
    this.aiService.pauseTraining();
  }

  resumeTraining(): void {
    this.aiService.resumeTraining();
  }

  stopTraining(): void {
    this._isDqnActive = false;
    this._networkStatusMessage = 'Network training stopped.';
    this.aiService.stopTraining();
  }

  testModel(): void {
    this.aiService.testModel();
  }

  saveModel(): void {
    this.aiService.saveModel();
  }

  loadModel(): void {
    this.aiService.loadModel();
  }

  getCurrentProgress(): number {
    return this.aiService.getCurrentProgress();
  }

  canTestModel(): boolean {
    const stats = this.trainingStatsSubject.value;
    // Allow testing if we have completed at least some training episodes
    return stats !== null && stats.episode > 0;
  }

  resetVisualizationState(): void {
    this._isDqnActive = false;
    this._networkStatusMessage = 'Initializing neural network...';
  }

  updateVisualizationInsights(): void {
    if (this._isDqnActive) {
      this._networkStatusMessage = 'Neural network is processing and learning from experience...';
    }
  }

  getTestResultMessage(): string {
    return 'DQN model test completed';
  }

  updateNetworkStatus(isActive: boolean, message?: string): void {
    this._isDqnActive = isActive;
    if (message) {
      this._networkStatusMessage = message;
    }
  }

  ngOnDestroy(): void {
    if (this.statsSubscription) {
      this.statsSubscription.unsubscribe();
    }
  }
}