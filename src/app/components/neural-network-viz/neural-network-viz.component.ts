import { Component, OnInit, OnDestroy, ElementRef, ViewChild, NgZone } from '@angular/core';
import { CommonModule } from '@angular/common';
import { MatCardModule } from '@angular/material/card';
import { MatButtonModule } from '@angular/material/button';
import { MatIconModule } from '@angular/material/icon';
import { MatTooltipModule } from '@angular/material/tooltip';
import { Subscription } from 'rxjs';
import * as d3 from 'd3';

import { AiService } from '../../services/ai.service';
import { NetworkVisualization, NetworkLayer, Neuron, Connection, NEATStats, NEATGenome } from '../../models/ai.model';
import { AlgorithmType } from '../../models/maze.model'; // For context

@Component({
  selector: 'app-neural-network-viz',
  standalone: true,
  imports: [
    CommonModule,
    MatCardModule,
    MatButtonModule,
    MatIconModule,
    MatTooltipModule
  ],
  templateUrl: './neural-network-viz.component.html',
  styleUrl: './neural-network-viz.component.css'
})
export class NeuralNetworkVizComponent implements OnInit, OnDestroy {
  @ViewChild('networkSvg', { static: false }) private networkSvgRef!: ElementRef<SVGElement>;
  @ViewChild('evolutionSvg', { static: false }) private evolutionSvgRef!: ElementRef<SVGElement>;
  @ViewChild('fitnessGraphSvg', { static: false }) private fitnessGraphSvgRef!: ElementRef<SVGElement>;
  @ViewChild('championSvg', { static: false }) private championSvgRef!: ElementRef<SVGElement>;
  
  private svg: any;
  private evolutionSvg: any;
  private fitnessGraphSvg: any;
  private championSvg: any;
  private width = 800;
  private height = 500;
  private margin = { top: 20, right: 150, bottom: 20, left: 150 };

  networkData: NetworkVisualization | null = null;
  private subscriptions: Subscription[] = [];
  
  // DQN properties
  isDqnActive = false;
  statusMessage = 'Network visualization will appear here when a DQN agent is active.';

  // NEAT properties
  isNeatActive = false;
  neatStats: NEATStats | null = null;
  championGenome: NEATGenome | null = null;
  private fitnessHistory: { generation: number; bestFitness: number; avgFitness: number }[] = [];
  private speciesColors: Map<number, string> = new Map();
  private readonly speciesColorPalette = [
    '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', 
    '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9'
  ];

  constructor(private aiService: AiService, private ngZone: NgZone) {}

  ngOnInit(): void {
    this.subscriptions.push(
      this.aiService.trainingStatus$.subscribe(status => {
        this.isDqnActive = status.isRunning; 
        if (!status.isRunning) {
            this.statusMessage = 'DQN training is not currently active. Start training a DQN agent to see the visualization.';
            this.clearVisualization();
        }
      }),
      // Subscribe to NEAT stats for evolutionary visualization
      this.aiService.neatStats$.subscribe(stats => {
        this.neatStats = stats;
        this.isNeatActive = stats !== null;
        if (stats) {
          this.updateFitnessHistory(stats);
          this.updateChampionGenome(stats);
          this.drawEvolutionVisualization();
          this.drawFitnessGraph();
          this.drawChampionNetwork();
        }
      })
    );
    
    this.createSampleNetwork();
  }

  ngOnDestroy(): void {
    this.subscriptions.forEach(sub => sub.unsubscribe());
  }

  // This method would be triggered by AiService to update the visualization
  updateNetworkVisualization(data: NetworkVisualization): void {
    this.networkData = data;
    this.drawNetwork();
  }
  
  private createSampleNetwork(): void {
    // Create a simple 2-4-3 network structure for placeholder
    const inputLayer: NetworkLayer = {
        type: 'input',
        neurons: Array(4).fill(null).map((_, i) => ({ 
            id: `i${i}`, value: Math.random(), bias: 0, activation: Math.random(), weights: [], connections: [] 
        }))
    };
    const hiddenLayer1: NetworkLayer = {
        type: 'hidden',
        neurons: Array(5).fill(null).map((_, i) => ({ 
            id: `h1_${i}`, value: Math.random(), bias: Math.random() * 0.1, activation: Math.random(), weights: [], connections: [] 
        }))
    };
    const outputLayer: NetworkLayer = {
        type: 'output',
        neurons: Array(4).fill(null).map((_, i) => ({ 
            id: `o${i}`, value: Math.random(), bias: 0, activation: Math.random(), weights: [], connections: [] 
        }))
    };

    const layers = [inputLayer, hiddenLayer1, outputLayer];
    const connections: Connection[] = [];

    for (let l = 0; l < layers.length - 1; l++) {
        layers[l].neurons.forEach(neuron1 => {
            layers[l+1].neurons.forEach(neuron2 => {
                connections.push({ 
                    from: neuron1.id, 
                    to: neuron2.id, 
                    weight: (Math.random() - 0.5) * 2, // Random weight between -1 and 1
                    active: Math.random() > 0.3 // Randomly active
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

        // Draw connections
        this.networkData!.connections.forEach(conn => {
            const sourceNeuronPos = layerPositions.flatMap(lp => lp.neurons).find(n => n.neuron.id === conn.from);
            const targetNeuronPos = layerPositions.flatMap(lp => lp.neurons).find(n => n.neuron.id === conn.to);

            if (sourceNeuronPos && targetNeuronPos) {
                this.svg.append('line')
                    .attr('x1', sourceNeuronPos.x)
                    .attr('y1', sourceNeuronPos.y)
                    .attr('x2', targetNeuronPos.x)
                    .attr('y2', targetNeuronPos.y)
                    .attr('stroke', conn.active ? (conn.weight > 0 ? '#2196F3' : '#FF6B6B') : '#ddd')
                    .attr('stroke-width', conn.active ? Math.abs(conn.weight * 2) + 0.5 : 0.5)
                    .attr('opacity', conn.active ? 0.8 : 0.3);
            }
        });
        
        // Draw neurons
        layerPositions.forEach((lp) => {
            lp.neurons.forEach(neuronPos => {
                this.svg.append('circle')
                    .attr('cx', neuronPos.x)
                    .attr('cy', neuronPos.y)
                    .attr('r', 10)
                    .attr('fill', this.getNeuronColor(neuronPos.neuron))
                    .attr('stroke', '#333')
                    .attr('stroke-width', 1.5)
                    .append('title')
                    .text(`ID: ${neuronPos.neuron.id}\nValue: ${neuronPos.neuron.value.toFixed(3)}\nActivation: ${neuronPos.neuron.activation.toFixed(3)}`);

                this.svg.append('text')
                    .attr('x', neuronPos.x)
                    .attr('y', neuronPos.y + 4) // Adjust for vertical centering
                    .attr('text-anchor', 'middle')
                    .attr('font-size', '9px')
                    .attr('fill', 'white')
                    .text(neuronPos.neuron.activation.toFixed(1));
            });
        });

        // Layer labels
        this.networkData!.layers.forEach((layer, i) => {
            this.svg.append('text')
                .attr('x', i * layerGap)
                .attr('y', chartHeight + this.margin.bottom / 2 + 10)
                .attr('text-anchor', 'middle')
                .attr('font-weight', 'bold')
                .text(layer.type.toUpperCase());
        });
    });
  }

  private getNeuronColor(neuron: Neuron & { type?: string }): string {
    const activation = neuron.activation;
    if (neuron.type === 'input') return d3.interpolateGreens(activation);
    if (neuron.type === 'output') return d3.interpolateBlues(activation);
    return d3.interpolateOranges(activation);
  }
  
  private clearVisualization(): void {
    if (this.networkSvgRef) {
        d3.select(this.networkSvgRef.nativeElement).selectAll('*').remove();
    }
  }
  
  refreshVisualization(): void {
    // This is a placeholder. Ideally, AiService would push new data.
    // For now, just redraw the sample or existing data.
    if (this.isDqnActive && !this.networkData) {
        this.createSampleNetwork();
    } else if (this.networkData) {
        this.drawNetwork();
    } else {
        this.statusMessage = 'No network data available to refresh.';
    }
  }

  // NEAT Visualization Methods
  private updateFitnessHistory(stats: NEATStats): void {
    const entry = {
      generation: stats.generation,
      bestFitness: stats.bestFitness,
      avgFitness: stats.averageFitness
    };
    
    // Only add if it's a new generation
    const lastEntry = this.fitnessHistory[this.fitnessHistory.length - 1];
    if (!lastEntry || lastEntry.generation !== stats.generation) {
      this.fitnessHistory.push(entry);
      // Keep only last 50 generations for performance
      if (this.fitnessHistory.length > 50) {
        this.fitnessHistory.shift();
      }
    }
  }

  private updateChampionGenome(stats: NEATStats): void {
    if (stats.topAgent && this.aiService.getNEATVisualizationData) {
      const neatData = this.aiService.getNEATVisualizationData();
      this.championGenome = neatData?.bestGenome || null;
    }
  }

  // Template helper methods
  trackSpecies(index: number, species: any): number {
    return species.id;
  }

  getSpeciesBubbleTransform(species: any): string {
    const angle = (species.id * 360 / (this.neatStats?.speciesCount || 1)) * (Math.PI / 180);
    const radius = 80 + (species.size * 2);
    const x = Math.cos(angle) * radius + 200;
    const y = Math.sin(angle) * radius + 150;
    return `translate(${x}px, ${y}px)`;
  }

  getSpeciesColor(speciesId: number): string {
    if (!this.speciesColors.has(speciesId)) {
      const colorIndex = speciesId % this.speciesColorPalette.length;
      this.speciesColors.set(speciesId, this.speciesColorPalette[colorIndex]);
    }
    return this.speciesColors.get(speciesId)!;
  }

  getFitnessBarHeight(species: any): number {
    const maxFitness = this.neatStats?.bestFitness || 1;
    return Math.min(100, (species.averageFitness / maxFitness) * 100);
  }

  private drawEvolutionVisualization(): void {
    if (!this.evolutionSvgRef || !this.neatStats) return;

    this.ngZone.runOutsideAngular(() => {
      d3.select(this.evolutionSvgRef.nativeElement).selectAll('*').remove();

      this.evolutionSvg = d3.select(this.evolutionSvgRef.nativeElement)
        .attr('width', 400)
        .attr('height', 300);

      // Draw evolutionary tree background
      this.evolutionSvg.append('circle')
        .attr('cx', 200)
        .attr('cy', 150)
        .attr('r', 120)
        .attr('fill', 'none')
        .attr('stroke', '#e0e0e0')
        .attr('stroke-width', 2)
        .attr('stroke-dasharray', '5,5');

      // Add generation indicator
      this.evolutionSvg.append('text')
        .attr('x', 200)
        .attr('y', 30)
        .attr('text-anchor', 'middle')
        .attr('font-size', '16px')
        .attr('font-weight', 'bold')
        .attr('fill', '#333')
        .text(`Generation ${this.neatStats?.generation ?? 0}`);

      // Add fitness indicator in center
      this.evolutionSvg.append('text')
        .attr('x', 200)
        .attr('y', 145)
        .attr('text-anchor', 'middle')
        .attr('font-size', '12px')
        .attr('fill', '#666')
        .text('Best Fitness');

      this.evolutionSvg.append('text')
        .attr('x', 200)
        .attr('y', 165)
        .attr('text-anchor', 'middle')
        .attr('font-size', '18px')
        .attr('font-weight', 'bold')
        .attr('fill', '#4CAF50')
        .text((this.neatStats?.bestFitness ?? 0).toFixed(1));
    });
  }

  private drawFitnessGraph(): void {
    if (!this.fitnessGraphSvgRef || this.fitnessHistory.length === 0) return;

    this.ngZone.runOutsideAngular(() => {
      d3.select(this.fitnessGraphSvgRef.nativeElement).selectAll('*').remove();

      const width = 400;
      const height = 200;
      const margin = { top: 20, right: 30, bottom: 30, left: 50 };

      this.fitnessGraphSvg = d3.select(this.fitnessGraphSvgRef.nativeElement)
        .attr('width', width)
        .attr('height', height);

      const xScale = d3.scaleLinear()
        .domain(d3.extent(this.fitnessHistory, d => d.generation) as [number, number])
        .range([margin.left, width - margin.right]);

      const yScale = d3.scaleLinear()
        .domain([0, d3.max(this.fitnessHistory, d => d.bestFitness) || 1])
        .range([height - margin.bottom, margin.top]);

      // Draw best fitness line
      const bestLine = d3.line<any>()
        .x(d => xScale(d.generation))
        .y(d => yScale(d.bestFitness))
        .curve(d3.curveMonotoneX);

      this.fitnessGraphSvg.append('path')
        .datum(this.fitnessHistory)
        .attr('fill', 'none')
        .attr('stroke', '#4CAF50')
        .attr('stroke-width', 3)
        .attr('d', bestLine);

      // Draw average fitness line
      const avgLine = d3.line<any>()
        .x(d => xScale(d.generation))
        .y(d => yScale(d.avgFitness))
        .curve(d3.curveMonotoneX);

      this.fitnessGraphSvg.append('path')
        .datum(this.fitnessHistory)
        .attr('fill', 'none')
        .attr('stroke', '#2196F3')
        .attr('stroke-width', 2)
        .attr('stroke-dasharray', '5,5')
        .attr('d', avgLine);

      // Add axes
      this.fitnessGraphSvg.append('g')
        .attr('transform', `translate(0,${height - margin.bottom})`)
        .call(d3.axisBottom(xScale));

      this.fitnessGraphSvg.append('g')
        .attr('transform', `translate(${margin.left},0)`)
        .call(d3.axisLeft(yScale));
    });
  }

  private drawChampionNetwork(): void {
    if (!this.championSvgRef || !this.championGenome) return;

    this.ngZone.runOutsideAngular(() => {
      d3.select(this.championSvgRef.nativeElement).selectAll('*').remove();

      const width = 400;
      const height = 200;

      this.championSvg = d3.select(this.championSvgRef.nativeElement)
        .attr('width', width)
        .attr('height', height);

      // Create a simplified network layout
      const nodes = this.championGenome.nodes || [];
      const connections = this.championGenome.connections || [];

      // Position nodes by type
      const inputNodes = nodes.filter(n => n.type === 'input');
      const hiddenNodes = nodes.filter(n => n.type === 'hidden');
      const outputNodes = nodes.filter(n => n.type === 'output');

      const nodePositions = new Map();

      // Position input nodes
      inputNodes.forEach((node, i) => {
        nodePositions.set(node.id, {
          x: 50,
          y: 50 + (i * (height - 100) / Math.max(1, inputNodes.length - 1))
        });
      });

      // Position output nodes
      outputNodes.forEach((node, i) => {
        nodePositions.set(node.id, {
          x: width - 50,
          y: 50 + (i * (height - 100) / Math.max(1, outputNodes.length - 1))
        });
      });

      // Position hidden nodes
      hiddenNodes.forEach((node, i) => {
        nodePositions.set(node.id, {
          x: 150 + (i * 100),
          y: height / 2 + (Math.random() - 0.5) * 60
        });
      });

      // Draw connections
      connections.filter(c => c.enabled).forEach(conn => {
        const from = nodePositions.get(conn.inputNode);
        const to = nodePositions.get(conn.outputNode);
        if (from && to) {
          this.championSvg.append('line')
            .attr('x1', from.x)
            .attr('y1', from.y)
            .attr('x2', to.x)
            .attr('y2', to.y)
            .attr('stroke', conn.weight > 0 ? '#4CAF50' : '#FF5722')
            .attr('stroke-width', Math.abs(conn.weight) * 2 + 0.5)
            .attr('opacity', 0.7);
        }
      });

      // Draw nodes
      nodes.forEach(node => {
        const pos = nodePositions.get(node.id);
        if (pos) {
          this.championSvg.append('circle')
            .attr('cx', pos.x)
            .attr('cy', pos.y)
            .attr('r', 8)
            .attr('fill', node.type === 'input' ? '#4CAF50' : 
                         node.type === 'output' ? '#2196F3' : '#FF9800')
            .attr('stroke', '#333')
            .attr('stroke-width', 1);
        }
      });
    });
  }

  resetNeatVisualization(): void {
    this.fitnessHistory = [];
    this.speciesColors.clear();
    this.championGenome = null;
    if (this.evolutionSvgRef) {
      d3.select(this.evolutionSvgRef.nativeElement).selectAll('*').remove();
    }
    if (this.fitnessGraphSvgRef) {
      d3.select(this.fitnessGraphSvgRef.nativeElement).selectAll('*').remove();
    }
    if (this.championSvgRef) {
      d3.select(this.championSvgRef.nativeElement).selectAll('*').remove();
    }
  }
}