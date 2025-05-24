import { Injectable, OnDestroy } from '@angular/core';
import { BaseAlgorithm, AlgorithmConfig } from './base-algorithm';
import { AlgorithmType } from '../../models/maze.model';
import { NEATConfig } from '../../models/ai.model';
import { AiService } from '../ai.service';
import { Subscription } from 'rxjs';

@Injectable({
  providedIn: 'root'
})
export class NEATAlgorithm extends BaseAlgorithm implements OnDestroy {
  private config: NEATConfig = {
    populationSize: 150,
    maxStagnation: 15,
    speciesThreshold: 3.0,
    survivalRate: 0.2,
    mutationRate: 0.8,
    crossoverRate: 0.75,
    maxStepsPerAgent: 200,
    excessCoefficient: 1.0,
    disjointCoefficient: 1.0,
    weightCoefficient: 0.4,
    addNodeMutationRate: 0.03,
    addConnectionMutationRate: 0.05,
    weightMutationRate: 0.8,
    weightPerturbationRate: 0.9
  };

  private _networkStatusMessage = 'Start NEAT training to see evolved neural networks.';
  private readonly neatStatsSubscription: Subscription;
  private readonly trainingStatusSubscription: Subscription;

  constructor(private readonly aiService: AiService) {
    super();
    // Subscribe to AI service NEAT stats and pass them through to our own subject
    this.neatStatsSubscription = this.aiService.neatStats$.subscribe(stats => {
      console.log('NEAT Algorithm received stats from AI service:', stats); // Debug log
      this.neatStatsSubject.next(stats);
      if (stats) {
        this.updateVisualizationInsights();
      }
    });

    // Subscribe to training status to update messages
    this.trainingStatusSubscription = this.aiService.trainingStatus$.subscribe(status => {
      if (status.isRunning) {
        if (status.isPaused) {
          this._networkStatusMessage = 'NEAT training paused...';
        } else {
          const stats = this.neatStatsSubject.value;
          if (stats && stats.generation > 0) {
            this._networkStatusMessage = `Generation ${stats.generation}: Evolving networks... Best fitness: ${stats.bestFitness.toFixed(2)}`;
          } else {
            this._networkStatusMessage = 'Starting NEAT evolution...';
          }
        }
      } else {
        if (status.message) {
          this._networkStatusMessage = status.message;
        } else {
          this._networkStatusMessage = 'NEAT training stopped.';
        }
      }
    });
  }

  get algorithmType(): AlgorithmType {
    return AlgorithmType.NEAT;
  }

  get qTableSize(): string {
    return 'N/A (Evolved Networks)';
  }

  get bestActionConfidence(): string {
    const stats = this.neatStatsSubject.value;
    return stats ? `${stats.bestFitness.toFixed(2)}` : 'N/A';
  }

  get networkStatusMessage(): string {
    return this._networkStatusMessage;
  }

  get isDqnActive(): boolean {
    return false; // NEAT doesn't use DQN
  }

  getDefaultConfig(): NEATConfig {
    return { ...this.config };
  }

  initializeConfiguration(algorithmConfig: AlgorithmConfig): void {
    if (algorithmConfig.neatConfig) {
      this.config = { ...algorithmConfig.neatConfig };
    }
  }

  startTraining(): void {
    this._networkStatusMessage = 'Initializing NEAT population...';
    this.aiService.startTraining(AlgorithmType.NEAT, this.config);
  }

  pauseTraining(): void {
    this.aiService.pauseTraining();
  }

  resumeTraining(): void {
    this.aiService.resumeTraining();
  }

  stopTraining(): void {
    this._networkStatusMessage = 'NEAT evolution stopped.';
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
    const stats = this.neatStatsSubject.value;
    return stats !== null && stats.generation > 0;
  }

  resetVisualizationState(): void {
    this._networkStatusMessage = 'Start NEAT training to see evolved neural networks.';
  }

  updateVisualizationInsights(): void {
    const stats = this.neatStatsSubject.value;
    if (stats) {
      if (stats.generation === 0) {
        this._networkStatusMessage = 'NEAT population initialized. Starting evolution...';
      } else {
        this._networkStatusMessage = `Generation ${stats.generation}: Best fitness ${stats.bestFitness.toFixed(2)}, Species: ${stats.speciesCount}`;
      }
    }
  }

  getTestResultMessage(): string {
    const stats = this.neatStatsSubject.value;
    if (stats) {
      return `NEAT evolution test completed. Best fitness: ${stats.bestFitness.toFixed(2)}`;
    }
    return 'NEAT model test completed';
  }

  updateNetworkStatus(message: string): void {
    this._networkStatusMessage = message;
  }

  ngOnDestroy(): void {
    if (this.neatStatsSubscription) {
      this.neatStatsSubscription.unsubscribe();
    }
    if (this.trainingStatusSubscription) {
      this.trainingStatusSubscription.unsubscribe();
    }
  }
}