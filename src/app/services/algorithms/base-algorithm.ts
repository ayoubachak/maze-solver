import { Observable, BehaviorSubject } from 'rxjs';
import { AlgorithmType } from '../../models/maze.model';
import { TrainingConfig, TrainingStats, NEATConfig, NEATStats, NeuralNetworkConfig } from '../../models/ai.model';

export interface AlgorithmConfig {
  type: AlgorithmType;
  trainingConfig?: TrainingConfig;
  neatConfig?: NEATConfig;
  neuralNetworkConfig?: NeuralNetworkConfig;
}

export abstract class BaseAlgorithm {
  protected trainingStatsSubject = new BehaviorSubject<TrainingStats | null>(null);
  protected neatStatsSubject = new BehaviorSubject<NEATStats | null>(null);
  
  trainingStats$ = this.trainingStatsSubject.asObservable();
  neatStats$ = this.neatStatsSubject.asObservable();

  abstract get algorithmType(): AlgorithmType;
  abstract get qTableSize(): string;
  abstract get bestActionConfidence(): string;
  abstract get networkStatusMessage(): string;
  abstract get isDqnActive(): boolean;

  abstract initializeConfiguration(config: AlgorithmConfig): void;
  abstract startTraining(): void;
  abstract pauseTraining(): void;
  abstract resumeTraining(): void;
  abstract stopTraining(): void;
  abstract testModel(): void;
  abstract saveModel(): void;
  abstract loadModel(): void;
  abstract getCurrentProgress(): number;
  abstract canTestModel(): boolean;
  abstract resetVisualizationState(): void;
  abstract updateVisualizationInsights(): void;
  abstract getTestResultMessage(): string;
}