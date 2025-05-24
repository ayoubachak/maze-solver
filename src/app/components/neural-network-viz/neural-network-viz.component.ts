import { Component, OnInit, OnDestroy, ElementRef, ViewChild, NgZone } from '@angular/core';
import { CommonModule } from '@angular/common';
import { MatCardModule } from '@angular/material/card';
import { MatButtonModule } from '@angular/material/button';
import { MatIconModule } from '@angular/material/icon';
import { Subscription } from 'rxjs';
import * as d3 from 'd3';

import { AiService } from '../../services/ai.service';
import { NetworkVisualization, NetworkLayer, Neuron, Connection } from '../../models/ai.model';
import { AlgorithmType } from '../../models/maze.model'; // For context

@Component({
  selector: 'app-neural-network-viz',
  standalone: true,
  imports: [
    CommonModule,
    MatCardModule,
    MatButtonModule,
    MatIconModule
  ],
  templateUrl: './neural-network-viz.component.html',
  styleUrl: './neural-network-viz.component.css'
})
export class NeuralNetworkVizComponent implements OnInit, OnDestroy {
  @ViewChild('networkSvg', { static: false }) private networkSvgRef!: ElementRef<SVGElement>;
  
  private svg: any;
  private width = 800;
  private height = 500;
  private margin = { top: 20, right: 150, bottom: 20, left: 150 };

  networkData: NetworkVisualization | null = null;
  private subscriptions: Subscription[] = [];
  
  isDqnActive = false;
  statusMessage = 'Network visualization will appear here when a DQN agent is active.';

  constructor(private aiService: AiService, private ngZone: NgZone) {}

  ngOnInit(): void {
    // Placeholder: In a real scenario, this would be dynamically updated
    // by the AiService when a DQN model is active and provides visualization data.
    this.subscriptions.push(
      this.aiService.trainingStatus$.subscribe(status => {
        // Simplified: Assume DQN is active if any AI training is running
        // A more robust check for AlgorithmType.DQN would be needed from AiService
        this.isDqnActive = status.isRunning; 
        if (!status.isRunning) {
            this.statusMessage = 'DQN training is not currently active. Start training a DQN agent to see the visualization.';
            this.clearVisualization();
        }
      }),
      // Placeholder for a dedicated network data observable from AiService
      // this.aiService.networkVisualization$.subscribe(data => { ...
    );
    
    // For now, let's create a sample network if no real data is coming
    // if (!this.networkData && this.isDqnActive) { 
    //    this.createSampleNetwork(); 
    // }
    this.createSampleNetwork(); // Draw a sample network initially for layout purposes
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
} 