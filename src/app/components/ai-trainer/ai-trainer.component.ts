import { Component, OnInit, OnDestroy, AfterViewInit, ViewChild, ElementRef, NgZone, ChangeDetectorRef } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { trigger, state, style, transition, animate } from '@angular/animations';
import { MatCardModule } from '@angular/material/card';
import { MatButtonModule } from '@angular/material/button';
import { MatIconModule } from '@angular/material/icon';
import { MatSelectModule } from '@angular/material/select';
import { MatFormFieldModule } from '@angular/material/form-field';
import { MatSliderModule } from '@angular/material/slider';
import { MatCheckboxModule } from '@angular/material/checkbox';
import { MatProgressBarModule } from '@angular/material/progress-bar';
import { MatInputModule } from '@angular/material/input';
import { MatTooltipModule } from '@angular/material/tooltip';
import { MatDividerModule } from '@angular/material/divider';
import { Subscription } from 'rxjs';
import * as d3 from 'd3';

import { AiService } from '../../services/ai.service';
import { MazeService } from '../../services/maze.service';
import { AlgorithmFactory } from '../../services/algorithms/algorithm-factory.service';
import { BaseAlgorithm } from '../../services/algorithms/base-algorithm';
import { QLearningAlgorithm } from '../../services/algorithms/qlearning-algorithm';
import { DQNAlgorithm } from '../../services/algorithms/dqn-algorithm';
import { NEATAlgorithm } from '../../services/algorithms/neat-algorithm';
import { Maze, CellType, AlgorithmType } from '../../models/maze.model';
import { TrainingConfig, NeuralNetworkConfig, TrainingStats, NetworkVisualization, NetworkLayer, Neuron, Connection, NEATConfig, NEATStats } from '../../models/ai.model';

@Component({
  selector: 'app-ai-trainer',
  standalone: true,
  imports: [
    CommonModule,
    FormsModule,
    MatCardModule,
    MatButtonModule,
    MatIconModule,
    MatSelectModule,
    MatFormFieldModule,
    MatSliderModule,
    MatCheckboxModule,
    MatProgressBarModule,
    MatInputModule,
    MatTooltipModule,
    MatDividerModule
  ],
  templateUrl: './ai-trainer.component.html',
  styleUrl: './ai-trainer.component.css',
  animations: [
    trigger('expandCollapse', [
      state('collapsed', style({ height: '0', opacity: '0', overflow: 'hidden' })),
      state('expanded', style({ height: '*', opacity: '1', overflow: 'visible' })),
      transition('collapsed <=> expanded', animate('300ms ease-in-out')),
    ])
  ]
})
export class AiTrainerComponent implements OnInit, OnDestroy, AfterViewInit {
  @ViewChild('networkSvg', { static: false }) private readonly networkSvgRef!: ElementRef<SVGElement>;
  
  maze: Maze | null = null;
  isRunning = false;
  isPaused = false;
  isTesting = false;
  trainingStats: TrainingStats | null = null;
  neatStats: NEATStats | null = null;
  testStats: any = null;
  currentAgentPosition: { x: number, y: number } | null = null;

  // Visualization controls
  showNetworkViz = false;
  showQValues = false;
  networkData: NetworkVisualization | null = null;

  // Enhanced visualization settings
  visualizationEnabled = true;
  visualizationSpeed = 50; // 1-100 scale
  showAgentPath = true;
  showExploredCells = true;
  enableRealTimeViz = true;
  maxPathLength = 100; // Limit path history for performance
  showVisualizationPanel = true; // New property for collapsible panel
  
  // Agent tracking
  agentPath: { x: number, y: number }[] = [];
  exploredCells = new Set<string>();
  lastAgentUpdate = 0;
  private readonly AGENT_UPDATE_THROTTLE = 50; // ms

  // Neural network visualization
  private svg: any;
  private width = 300;
  private height = 200;
  private readonly margin = { top: 10, right: 10, bottom: 20, left: 10 };

  selectedAlgorithm: AlgorithmType = AlgorithmType.QLEARNING;
  currentAlgorithm: BaseAlgorithm;
  algorithms: { value: AlgorithmType; name: string; description: string; algorithm: BaseAlgorithm }[] = [];

  // Algorithm configurations - now using the algorithm classes
  qLearningConfig: TrainingConfig;
  dqnTrainingConfig: TrainingConfig;
  dqnNetworkConfig: NeuralNetworkConfig;
  neatConfig: NEATConfig;
  
  // Maze settings
  showMazeSettings = false;
  
  mazeSize = { width: 15, height: 11 }; // Smaller maze for faster training
  CellType = CellType;
  AlgorithmType = AlgorithmType;
  
  private readonly subscriptions: Subscription[] = [];

  constructor(
    private readonly mazeService: MazeService,
    private readonly aiService: AiService,
    private readonly algorithmFactory: AlgorithmFactory,
    private readonly cdr: ChangeDetectorRef,
    private readonly ngZone: NgZone
  ) {
    // Initialize algorithms and configurations
    this.algorithms = this.algorithmFactory.getAllAlgorithms();
    this.currentAlgorithm = this.algorithms[0].algorithm;
    
    // Get default configurations from algorithm classes
    this.qLearningConfig = (this.algorithmFactory.getAlgorithm(AlgorithmType.QLEARNING) as QLearningAlgorithm).getDefaultConfig();
    
    const dqnAlgorithm = this.algorithmFactory.getAlgorithm(AlgorithmType.DQN) as DQNAlgorithm;
    this.dqnTrainingConfig = dqnAlgorithm.getDefaultTrainingConfig();
    this.dqnNetworkConfig = dqnAlgorithm.getDefaultNetworkConfig();
    
    this.neatConfig = (this.algorithmFactory.getAlgorithm(AlgorithmType.NEAT) as NEATAlgorithm).getDefaultConfig();
  }

  ngOnInit(): void {
    this.subscriptions.push(
      this.mazeService.currentMaze$.subscribe(maze => {
        this.maze = maze;
        if (maze) {
          this.aiService.initializeEnvironment(maze);
        }
        this.cdr.detectChanges();
      }),
      this.aiService.trainingStats$.subscribe(stats => {
        this.ngZone.run(() => {
          this.trainingStats = stats;
          this.currentAlgorithm.updateVisualizationInsights();
          this.cdr.detectChanges();
        });
      }),
      this.aiService.agentMoved$.subscribe(position => {
        this.ngZone.run(() => {
          this.currentAgentPosition = position;
          if (this.maze && position) {
            this.updateAgentVisualization(position);
          }
          this.cdr.detectChanges(); 
        });
      }),
      this.aiService.trainingStatus$.subscribe(status => {
        this.isRunning = status.isRunning;
        this.isPaused = status.isPaused;
        
        // Update algorithm-specific status
        if (this.currentAlgorithm.algorithmType === AlgorithmType.DQN) {
          const dqnAlgorithm = this.currentAlgorithm as DQNAlgorithm;
          dqnAlgorithm.updateNetworkStatus(status.isRunning, 
            status.isRunning ? 'Neural network is active and learning...' : 'Start DQN training to see neural network activity.'
          );
          
          if (status.isRunning && this.showNetworkViz) {
            this.createSampleNetwork();
          }
        } else if (this.currentAlgorithm.algorithmType === AlgorithmType.NEAT) {
          const neatAlgorithm = this.currentAlgorithm as NEATAlgorithm;
          neatAlgorithm.updateNetworkStatus(
            status.isRunning ? 'NEAT networks are evolving...' : 'Start NEAT training to see evolved neural networks.'
          );
          
          // Create NEAT network visualization when training starts, even without stats initially
          if (status.isRunning && this.showNetworkViz) {
            // Create initial visualization or wait for first stats
            setTimeout(() => {
              if (this.neatStats) {
                this.createNEATNetworkVisualization();
              } else {
                // Create a placeholder visualization for generation 0
                this.createInitialNEATVisualization();
              }
            }, 100);
          }
        }
        
        if (!status.isRunning && status.message) {
          console.log("Training status: ", status.message);
        }
        
        this.cdr.detectChanges();
      }),
      // Subscribe to testing status
      this.aiService.testingStatus$.subscribe(status => {
        this.ngZone.run(() => {
          this.isTesting = status.isRunning;
          this.cdr.detectChanges();
        });
      }),
      // Subscribe to test results
      this.aiService.testStats$.subscribe(stats => {
        this.ngZone.run(() => {
          this.testStats = stats;
          this.cdr.detectChanges();
        });
      }),
      // Subscribe to NEAT stats
      this.aiService.neatStats$.subscribe(stats => {
        this.ngZone.run(() => {
          this.neatStats = stats;
          if (this.currentAlgorithm && this.currentAlgorithm.algorithmType === AlgorithmType.NEAT) {
            this.currentAlgorithm.updateVisualizationInsights();
            // Update NEAT network visualization if enabled
            if (this.showNetworkViz && stats) {
              this.createNEATNetworkVisualization();
            }
          }
          this.cdr.detectChanges();
        });
      })
    );
    this.generateMaze(); 
  }

  ngAfterViewInit(): void {
    // Initialize network visualization if needed
    if (this.selectedAlgorithm === AlgorithmType.DQN && this.showNetworkViz) {
      setTimeout(() => this.initializeNetworkVisualization(), 100);
    }
  }

  ngOnDestroy(): void {
    this.subscriptions.forEach(sub => sub.unsubscribe());
    this.aiService.stopTraining();
  }

  // Delegate methods to current algorithm
  get qTableSize(): string {
    return this.currentAlgorithm.qTableSize;
  }

  get bestActionConfidence(): string {
    return this.currentAlgorithm.bestActionConfidence;
  }

  get networkStatusMessage(): string {
    return this.currentAlgorithm.networkStatusMessage;
  }

  get isDqnActive(): boolean {
    return this.currentAlgorithm.isDqnActive;
  }

  generateMaze(): void {
    this.currentAlgorithm.stopTraining();
    this.mazeService.generateMaze(this.mazeSize.width, this.mazeSize.height);
  }

  startTraining(): void {
    if (!this.maze) return;
    
    // Reset visualization state when starting new training
    this.resetVisualizationState();
    
    // Configure the algorithm with current settings
    this.currentAlgorithm.initializeConfiguration({
      type: this.selectedAlgorithm,
      trainingConfig: this.selectedAlgorithm === AlgorithmType.QLEARNING ? this.qLearningConfig : this.dqnTrainingConfig,
      neatConfig: this.selectedAlgorithm === AlgorithmType.NEAT ? this.neatConfig : undefined,
      neuralNetworkConfig: this.selectedAlgorithm === AlgorithmType.DQN ? this.dqnNetworkConfig : undefined
    });
    
    this.currentAlgorithm.startTraining();
  }

  pauseTraining(): void {
    this.currentAlgorithm.pauseTraining();
  }

  resumeTraining(): void {
    this.currentAlgorithm.resumeTraining();
  }

  stopTraining(): void {
    this.currentAlgorithm.stopTraining();
  }

  testModel(): void {
    if (!this.maze) return;
    this.currentAlgorithm.testModel();
  }

  saveModel(): void {
    this.currentAlgorithm.saveModel();
  }

  loadModel(): void {
    this.currentAlgorithm.loadModel();
  }

  getTrainingProgress(): number {
    return this.currentAlgorithm.getCurrentProgress();
  }

  canTestModel(): boolean {
    return !this.isRunning && this.currentAlgorithm.canTestModel();
  }

  canPauseResume(): boolean {
    return this.isRunning;
  }

  getTestResultMessage(): string {
    if (!this.testStats) return '';
    
    if (this.testStats.success) {
      return `✅ Success! Reached goal in ${this.testStats.totalSteps} steps with reward ${this.testStats.reward.toFixed(1)}`;
    } else {
      return `❌ Failed to reach goal after ${this.testStats.totalSteps} steps. Reward: ${this.testStats.reward.toFixed(1)}`;
    }
  }

  getCellClass(cellType: CellType, x: number, y: number): string {
    const classes = ['maze-cell'];
    if (this.currentAgentPosition && this.currentAgentPosition.x === x && this.currentAgentPosition.y === y) {
      classes.push('current');
    } else {
      switch (cellType) {
        case CellType.WALL: classes.push('wall'); break;
        case CellType.START: classes.push('start'); break;
        case CellType.END: classes.push('end'); break;
        case CellType.PATH: classes.push('path'); break;
        case CellType.VISITED: classes.push('visited'); break;
        default: classes.push('empty');
      }
    }
    return classes.join(' ');
  }

  getCellTooltip(cellType: CellType, x: number, y: number): string {
    switch (cellType) {
      case CellType.START: return 'Start Position';
      case CellType.END: return 'Goal Position';
      case CellType.WALL: return 'Wall';
      case CellType.CURRENT: return 'Agent Current Position';
      case CellType.VISITED: return 'Visited Cell';
      case CellType.PATH: return 'Solution Path';
      default: return `Empty Cell (${x}, ${y})`;
    }
  }

  getAlgorithmDescription(): string {
    return this.algorithms.find(a => a.value === this.selectedAlgorithm)?.description ?? '';
  }

  // Maze configuration methods
  toggleMazeSettings(): void {
    this.showMazeSettings = !this.showMazeSettings;
  }

  setMazeSize(width: number, height: number): void {
    this.mazeSize = { width, height };
    // Automatically generate a new maze with the new dimensions
    this.generateMaze();
  }

  // Neural Network Visualization Methods
  toggleNetworkVisualization(): void {
    this.showNetworkViz = !this.showNetworkViz;
    if (this.showNetworkViz && this.selectedAlgorithm === AlgorithmType.DQN) {
      setTimeout(() => this.initializeNetworkVisualization(), 100);
    }
  }

  toggleQValues(): void {
    this.showQValues = !this.showQValues;
    // Here you could implement Q-value overlay visualization
  }

  private updateAgentPosition(position: { x: number, y: number }): void {
    if (!this.maze) return;
    
    // Clear previous agent position
    for (const row of this.maze.grid) {
      for (const cell of row) {
        if (cell.type === CellType.CURRENT) {
          cell.type = CellType.EMPTY;
        }
      }
    }
    
    // Mark new agent position
    const targetCell = this.maze.grid[position.y][position.x];
    if (targetCell.type !== CellType.END && targetCell.type !== CellType.START) {
      targetCell.type = CellType.CURRENT;
    }
  }

  private initializeNetworkVisualization(): void {
    if (!this.networkSvgRef) return;
    
    // Adjust dimensions for mobile
    const container = this.networkSvgRef.nativeElement.parentElement;
    if (container) {
      this.width = Math.min(container.clientWidth - 20, 400);
      this.height = Math.min(250, this.width * 0.6);
    }
    
    this.createSampleNetwork();
  }

  private createSampleNetwork(): void {
    if (!this.networkSvgRef) return;

    // Create a sample network structure based on DQN config
    const hiddenLayers = Array.isArray(this.dqnNetworkConfig.hiddenLayers) 
      ? this.dqnNetworkConfig.hiddenLayers 
      : String(this.dqnNetworkConfig.hiddenLayers).split(',').map((n: string) => parseInt(n.trim(), 10));

    const inputLayer: NetworkLayer = {
      type: 'input',
      neurons: Array(4).fill(null).map((_, i) => ({ 
        id: `i${i}`, 
        value: Math.random(), 
        bias: 0, 
        activation: Math.random(), 
        weights: [], 
        connections: [] 
      }))
    };

    const layers = [inputLayer];
    
    // Add hidden layers
    hiddenLayers.forEach((size: number, idx: number) => {
      const layerSize = Math.min(size, 8); // Limit visual complexity
      const hiddenLayer: NetworkLayer = {
        type: 'hidden',
        neurons: Array(layerSize).fill(null).map((_, i) => ({ 
          id: `h${idx}_${i}`, 
          value: Math.random(), 
          bias: Math.random() * 0.1, 
          activation: Math.random(), 
          weights: [], 
          connections: [] 
        }))
      };
      layers.push(hiddenLayer);
    });

    // Output layer (4 actions: up, down, left, right)
    const outputLayer: NetworkLayer = {
      type: 'output',
      neurons: Array(4).fill(null).map((_, i) => ({ 
        id: `o${i}`, 
        value: Math.random(), 
        bias: 0, 
        activation: Math.random(), 
        weights: [], 
        connections: [] 
      }))
    };
    layers.push(outputLayer);

    // Create connections
    const connections: Connection[] = [];
    for (let l = 0; l < layers.length - 1; l++) {
      layers[l].neurons.forEach(neuron1 => {
        layers[l+1].neurons.forEach(neuron2 => {
          connections.push({ 
            from: neuron1.id, 
            to: neuron2.id, 
            weight: (Math.random() - 0.5) * 2,
            active: Math.random() > 0.3
          });
        });
      });
    }
    
    this.networkData = {
      layers,
      connections,
      currentInput: inputLayer.neurons.map(n => n.value),
      currentOutput: outputLayer.neurons.map(n => n.value)
    };
    
    this.drawNetwork();
  }

  private drawNetwork(): void {
    if (!this.networkData || !this.networkSvgRef) return;
    
    this.ngZone.runOutsideAngular(() => {
      d3.select(this.networkSvgRef.nativeElement).selectAll('*').remove();

      this.svg = d3.select(this.networkSvgRef.nativeElement)
        .attr('width', this.width)
        .attr('height', this.height)
        .append('g')
        .attr('transform', `translate(${this.margin.left},${this.margin.top})`);

      const chartWidth = this.width - this.margin.left - this.margin.right;
      const chartHeight = this.height - this.margin.top - this.margin.bottom;
      const layerGap = chartWidth / (this.networkData!.layers.length - 1);

      // Calculate neuron positions
      const layerPositions: { layer: NetworkLayer, neurons: {neuron: Neuron, x: number, y: number}[] }[] = [];

      this.networkData!.layers.forEach((layer, i) => {
        const neuronsWithPositions: {neuron: Neuron, x: number, y: number}[] = [];
        const numNeurons = layer.neurons.length;
        const yGap = chartHeight / (numNeurons + 1);
        
        layer.neurons.forEach((neuron, j) => {
          neuronsWithPositions.push({
            neuron,
            x: i * layerGap,
            y: (j + 1) * yGap
          });
        });
        layerPositions.push({ layer, neurons: neuronsWithPositions });
      });

      // Draw connections (simplified for mobile)
      this.networkData!.connections.forEach(conn => {
        const sourceNeuronPos = layerPositions.flatMap(lp => lp.neurons).find(n => n.neuron.id === conn.from);
        const targetNeuronPos = layerPositions.flatMap(lp => lp.neurons).find(n => n.neuron.id === conn.to);

        if (sourceNeuronPos && targetNeuronPos && conn.active) {
          this.svg.append('line')
            .attr('x1', sourceNeuronPos.x)
            .attr('y1', sourceNeuronPos.y)
            .attr('x2', targetNeuronPos.x)
            .attr('y2', targetNeuronPos.y)
            .attr('stroke', conn.weight > 0 ? '#2196F3' : '#FF6B6B')
            .attr('stroke-width', Math.abs(conn.weight) + 0.5)
            .attr('opacity', 0.6);
        }
      });
      
      // Draw neurons
      layerPositions.forEach((lp) => {
        lp.neurons.forEach(neuronPos => {
          this.svg.append('circle')
            .attr('cx', neuronPos.x)
            .attr('cy', neuronPos.y)
            .attr('r', 6)
            .attr('fill', this.getNeuronColor(neuronPos.neuron, lp.layer.type))
            .attr('stroke', '#333')
            .attr('stroke-width', 1)
            .append('title')
            .text(`${lp.layer.type}: ${neuronPos.neuron.activation.toFixed(2)}`);
        });
      });
    });
  }

  private getNeuronColor(neuron: Neuron, layerType: string): string {
    const activation = Math.max(0, Math.min(1, neuron.activation));
    switch (layerType) {
      case 'input': return d3.interpolateGreens(0.3 + activation * 0.7);
      case 'output': return d3.interpolateBlues(0.3 + activation * 0.7);
      default: return d3.interpolateOranges(0.3 + activation * 0.7);
    }
  }

  private updateVisualizationInsights(): void {
    // Update neural network if active
    if (this.selectedAlgorithm === AlgorithmType.DQN && this.showNetworkViz && this.isDqnActive) {
      // Update network visualization with new data
      if (this.networkData) {
        // Simulate network activity changes
        this.networkData.layers.forEach(layer => {
          layer.neurons.forEach(neuron => {
            neuron.activation = Math.max(0, Math.min(1, neuron.activation + (Math.random() - 0.5) * 0.1));
          });
        });
        this.drawNetwork();
      }
    }
    
    // Update NEAT network visualization if active
    if (this.selectedAlgorithm === AlgorithmType.NEAT && this.showNetworkViz && this.isRunning) {
      // Update NEAT network visualization with evolving topology
      if (this.neatStats && this.networkData) {
        // Simulate evolution by updating neuron activations
        this.networkData.layers.forEach(layer => {
          layer.neurons.forEach(neuron => {
            neuron.activation = Math.max(0, Math.min(1, neuron.activation + (Math.random() - 0.5) * 0.15));
          });
        });
        this.drawNetwork();
      } else if (this.neatStats) {
        // Create new network visualization if needed
        this.createNEATNetworkVisualization();
      }
    }
  }

  // Enhanced visualization methods
  toggleVisualization(): void {
    this.visualizationEnabled = !this.visualizationEnabled;
    this.aiService.setOptimizedVisualization(!this.visualizationEnabled);
    
    if (!this.visualizationEnabled) {
      // Clear visual elements when disabled
      this.clearVisualizationElements();
    }
  }

  onVisualizationSpeedChange(event: any): void {
    this.visualizationSpeed = event.target?.value ?? event.value;
    this.aiService.setVisualizationSpeed(this.visualizationSpeed);
  }

  toggleRealTimeVisualization(): void {
    this.enableRealTimeViz = !this.enableRealTimeViz;
    // Additional logic for real-time updates
  }

  toggleAgentPath(): void {
    this.showAgentPath = !this.showAgentPath;
    if (!this.showAgentPath) {
      this.agentPath = [];
    }
  }

  toggleExploredCells(): void {
    this.showExploredCells = !this.showExploredCells;
    if (!this.showExploredCells) {
      this.exploredCells.clear();
    }
  }

  toggleVisualizationPanel(): void {
    this.showVisualizationPanel = !this.showVisualizationPanel;
  }

  private updateAgentVisualization(position: { x: number, y: number }): void {
    if (!this.visualizationEnabled || !this.enableRealTimeViz) return;

    const now = Date.now();
    if (now - this.lastAgentUpdate < this.AGENT_UPDATE_THROTTLE) {
      return; // Throttle updates
    }
    this.lastAgentUpdate = now;

    // Update agent path
    if (this.showAgentPath) {
      this.agentPath.push({ ...position });
      if (this.agentPath.length > this.maxPathLength) {
        this.agentPath.shift(); // Remove oldest position
      }
    }

    // Update explored cells
    if (this.showExploredCells) {
      const cellKey = `${position.x},${position.y}`;
      this.exploredCells.add(cellKey);
    }

    // Update visual position with optimized rendering
    if (this.enableRealTimeViz) {
      this.ngZone.runOutsideAngular(() => {
        this.updateAgentPosition(position);
        this.ngZone.run(() => this.cdr.detectChanges());
      });
    }
  }

  private clearVisualizationElements(): void {
    this.agentPath = [];
    this.exploredCells.clear();
    this.currentAgentPosition = null;
  }

  // Utility methods for visualization settings
  getVisualizationSpeedLabel(): string {
    if (this.visualizationSpeed <= 20) return 'Slow';
    if (this.visualizationSpeed <= 50) return 'Medium';
    if (this.visualizationSpeed <= 80) return 'Fast';
    return 'Very Fast';
  }

  isPathCellHighlighted(x: number, y: number): boolean {
    if (!this.showAgentPath) return false;
    return this.agentPath.some(pos => pos.x === x && pos.y === y);
  }

  isExploredCellHighlighted(x: number, y: number): boolean {
    if (!this.showExploredCells) return false;
    return this.exploredCells.has(`${x},${y}`);
  }

  getEnhancedCellClass(cellType: CellType, x: number, y: number): string {
    const classes = ['maze-cell'];
    
    // Current agent position takes priority
    if (this.currentAgentPosition && this.currentAgentPosition.x === x && this.currentAgentPosition.y === y) {
      classes.push('current');
    } else {
      // Base cell type
      switch (cellType) {
        case CellType.WALL: classes.push('wall'); break;
        case CellType.START: classes.push('start'); break;
        case CellType.END: classes.push('end'); break;
        case CellType.PATH: classes.push('path'); break;
        case CellType.VISITED: classes.push('visited'); break;
        default: classes.push('empty');
      }
      
      // Add visualization overlays if enabled
      if (this.visualizationEnabled) {
        if (this.isPathCellHighlighted(x, y)) {
          classes.push('agent-path');
        }
        if (this.isExploredCellHighlighted(x, y)) {
          classes.push('explored');
        }
      }
    }
    
    return classes.join(' ');
  }

  private resetVisualizationState(): void {
    // Clear all visualization data
    this.agentPath = [];
    this.exploredCells.clear();
    this.currentAgentPosition = null;
    this.testStats = null;
    
    // Reset network visualization
    this.networkData = null;
    
    // Clear any visual elements
    this.clearVisualizationElements();
  }

  onAlgorithmChange(): void {
    // Stop current training if running
    if (this.isRunning) {
      this.stopTraining();
    }
    
    // Clear all stats immediately to prevent showing old data
    this.trainingStats = null;
    this.neatStats = null;
    this.testStats = null;
    
    // Switch to the new algorithm
    this.currentAlgorithm = this.algorithmFactory.getAlgorithm(this.selectedAlgorithm);
    
    // Reset all stats and visualization immediately
    this.currentAlgorithm.resetVisualizationState();
    this.resetVisualizationState();
    
    // Update network visualization status based on new algorithm
    if (this.selectedAlgorithm === AlgorithmType.DQN) {
      this.showNetworkViz = false; // Reset network viz state
    } else if (this.selectedAlgorithm === AlgorithmType.NEAT) {
      this.showNetworkViz = false; // Reset network viz state
    } else {
      this.showNetworkViz = false; // Hide network viz for Q-Learning
    }
    
    // Force immediate change detection to update the UI
    this.cdr.markForCheck();
    this.cdr.detectChanges();
  }

  private createNEATNetworkVisualization(): void {
    if (!this.networkSvgRef || !this.neatStats) return;

    // Create a simplified visualization of the best NEAT genome
    // For now, create a sample network that represents the evolved topology
    const inputLayer: NetworkLayer = {
      type: 'input',
      neurons: Array(4).fill(null).map((_, i) => ({ 
        id: `i${i}`, 
        value: Math.random(), 
        bias: 0, 
        activation: Math.random(), 
        weights: [], 
        connections: [] 
      }))
    };

    // NEAT networks can have variable hidden layer structures
    // For visualization, we'll create a representation based on the best genome
    const hiddenNeurons = Math.min(Math.max(2, Math.floor(this.neatStats.generation / 5)), 6);
    const hiddenLayer: NetworkLayer = {
      type: 'hidden',
      neurons: Array(hiddenNeurons).fill(null).map((_, i) => ({ 
        id: `h${i}`, 
        value: Math.random(), 
        bias: Math.random() * 0.2 - 0.1, 
        activation: Math.random(), 
        weights: [], 
        connections: [] 
      }))
    };

    const outputLayer: NetworkLayer = {
      type: 'output',
      neurons: Array(4).fill(null).map((_, i) => ({ 
        id: `o${i}`, 
        value: Math.random(), 
        bias: 0, 
        activation: Math.random(), 
        weights: [], 
        connections: [] 
      }))
    };

    const layers = [inputLayer, hiddenLayer, outputLayer];

    // Create evolved connections (NEAT networks can have skip connections)
    const connections: Connection[] = [];
    
    // Input to hidden connections
    inputLayer.neurons.forEach(neuron1 => {
      hiddenLayer.neurons.forEach(neuron2 => {
        if (Math.random() > 0.3) { // Not all connections exist in NEAT
          connections.push({ 
            from: neuron1.id, 
            to: neuron2.id, 
            weight: (Math.random() - 0.5) * 3,
            active: true
          });
        }
      });
    });

    // Hidden to output connections
    hiddenLayer.neurons.forEach(neuron1 => {
      outputLayer.neurons.forEach(neuron2 => {
        connections.push({ 
          from: neuron1.id, 
          to: neuron2.id, 
          weight: (Math.random() - 0.5) * 2,
          active: true
        });
      });
    });

    // Skip connections (input to output) - characteristic of NEAT
    inputLayer.neurons.forEach(neuron1 => {
      outputLayer.neurons.forEach(neuron2 => {
        if (Math.random() > 0.7) { // Occasional skip connections
          connections.push({ 
            from: neuron1.id, 
            to: neuron2.id, 
            weight: (Math.random() - 0.5) * 1.5,
            active: true
          });
        }
      });
    });
    
    this.networkData = {
      layers,
      connections,
      currentInput: inputLayer.neurons.map(n => n.value),
      currentOutput: outputLayer.neurons.map(n => n.value)
    };
    
    this.drawNetwork();
  }

  private createInitialNEATVisualization(): void {
    if (!this.networkSvgRef) return;

    // Create a minimal initial visualization for generation 0
    const inputLayer: NetworkLayer = {
      type: 'input',
      neurons: Array(4).fill(null).map((_, i) => ({ 
        id: `i${i}`, 
        value: Math.random(), 
        bias: 0, 
        activation: Math.random(), 
        weights: [], 
        connections: [] 
      }))
    };

    // Start with a simple hidden layer structure
    const hiddenLayer: NetworkLayer = {
      type: 'hidden',
      neurons: Array(2).fill(null).map((_, i) => ({ 
        id: `h${i}`, 
        value: Math.random(), 
        bias: Math.random() * 0.2 - 0.1, 
        activation: Math.random(), 
        weights: [], 
        connections: [] 
      }))
    };

    const outputLayer: NetworkLayer = {
      type: 'output',
      neurons: Array(4).fill(null).map((_, i) => ({ 
        id: `o${i}`, 
        value: Math.random(), 
        bias: 0, 
        activation: Math.random(), 
        weights: [], 
        connections: [] 
      }))
    };

    const layers = [inputLayer, hiddenLayer, outputLayer];

    // Create basic connections for initial topology
    const connections: Connection[] = [];
    
    // Direct input to output connections (minimal NEAT topology)
    inputLayer.neurons.forEach(neuron1 => {
      outputLayer.neurons.forEach(neuron2 => {
        connections.push({ 
          from: neuron1.id, 
          to: neuron2.id, 
          weight: (Math.random() - 0.5) * 1,
          active: true
        });
      });
    });
    
    this.networkData = {
      layers,
      connections,
      currentInput: inputLayer.neurons.map(n => n.value),
      currentOutput: outputLayer.neurons.map(n => n.value)
    };
    
    this.drawNetwork();
  }
}