import { Injectable, OnDestroy } from '@angular/core';
import { BaseAlgorithm, AlgorithmConfig } from './base-algorithm';
import { AlgorithmType } from '../../models/maze.model';
import { TrainingConfig } from '../../models/ai.model';
import { AiService } from '../ai.service';
import { Subscription } from 'rxjs';

@Injectable({
  providedIn: 'root'
})
export class QLearningAlgorithm extends BaseAlgorithm implements OnDestroy {
  private config: TrainingConfig = {
    learningRate: 0.1,
    discountFactor: 0.9,
    explorationRate: 1.0,
    explorationDecay: 0.995,
    minExplorationRate: 0.01,
    episodes: 1000,
    maxStepsPerEpisode: 200
  };

  private _qTableSize = 'N/A';
  private _bestActionConfidence = 'N/A';
  private readonly _networkStatusMessage = 'Q-Learning uses a Q-table, not neural networks.';

  private readonly statsSubscription?: Subscription;

  constructor(private readonly aiService: AiService) {
    super();
    // Subscribe to AI service training stats
    this.statsSubscription = this.aiService.trainingStats$.subscribe(stats => {
      this.trainingStatsSubject.next(stats);
    });
  }

  get algorithmType(): AlgorithmType {
    return AlgorithmType.QLEARNING;
  }

  get qTableSize(): string {
    return this._qTableSize;
  }

  get bestActionConfidence(): string {
    return this._bestActionConfidence;
  }

  get networkStatusMessage(): string {
    return this._networkStatusMessage;
  }

  get isDqnActive(): boolean {
    return false;
  }

  getDefaultConfig(): TrainingConfig {
    return { ...this.config };
  }

  initializeConfiguration(algorithmConfig: AlgorithmConfig): void {
    if (algorithmConfig.trainingConfig) {
      this.config = { ...algorithmConfig.trainingConfig };
    }
  }

  startTraining(): void {
    this.aiService.startTraining(AlgorithmType.QLEARNING, this.config);
  }

  pauseTraining(): void {
    this.aiService.pauseTraining();
  }

  resumeTraining(): void {
    this.aiService.resumeTraining();
  }

  stopTraining(): void {
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
    this._qTableSize = 'N/A';
    this._bestActionConfidence = 'N/A';
  }

  updateVisualizationInsights(): void {
    const stats = this.trainingStatsSubject.value;
    if (stats && stats.episode > 0) {
      // Get real Q-table data from AI service
      const qTableArray = this.aiService.getQTableArray();
      
      // Calculate actual Q-table size
      const qTableSize = qTableArray.length;
      this._qTableSize = qTableSize > 0 ? `${qTableSize} states` : 'Empty';
      
      // Calculate best action confidence based on Q-values
      if (qTableSize > 0) {
        let totalConfidence = 0;
        let validStates = 0;
        
        qTableArray.forEach(entry => {
          const qValues = entry.qValues;
          const maxQ = Math.max(...qValues);
          const minQ = Math.min(...qValues);
          const range = maxQ - minQ;
          
          if (range > 0) {
            // Calculate confidence as the difference between best and second-best actions
            const sortedQValues = [...qValues].sort((a, b) => b - a);
            const confidence = (sortedQValues[0] - sortedQValues[1]) / range;
            totalConfidence += Math.max(0, Math.min(1, confidence));
            validStates++;
          }
        });
        
        if (validStates > 0) {
          const avgConfidence = totalConfidence / validStates;
          this._bestActionConfidence = `${(avgConfidence * 100).toFixed(1)}%`;
        } else {
          this._bestActionConfidence = '0%';
        }
      } else {
        this._bestActionConfidence = 'N/A';
      }
    }
  }

  getTestResultMessage(): string {
    // Implementation for Q-Learning specific test result message
    return 'Q-Learning model test completed';
  }

  ngOnDestroy(): void {
    if (this.statsSubscription) {
      this.statsSubscription.unsubscribe();
    }
  }
}