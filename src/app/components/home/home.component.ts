import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { RouterLink } from '@angular/router';
import { MatCardModule } from '@angular/material/card';
import { MatButtonModule } from '@angular/material/button';
import { MatIconModule } from '@angular/material/icon';

@Component({
  selector: 'app-home',
  standalone: true,
  imports: [
    CommonModule,
    RouterLink,
    MatCardModule,
    MatButtonModule,
    MatIconModule
  ],
  templateUrl: './home.component.html',
  styleUrl: './home.component.css'
})
export class HomeComponent {
  features = [
    {
      title: 'Maze Solver',
      description: 'Watch classic algorithms like A*, Dijkstra, and BFS solve mazes step by step',
      icon: 'extension',
      route: '/maze-solver',
      color: '#4CAF50'
    },
    {
      title: 'AI Trainer',
      description: 'Train a reinforcement learning agent to solve mazes using Q-Learning and Deep Q-Networks',
      icon: 'school',
      route: '/ai-trainer',
      color: '#2196F3'
    },
    {
      title: 'Neural Network',
      description: 'Visualize how the neural network learns and makes decisions in real-time',
      icon: 'device_hub',
      route: '/neural-network',
      color: '#FF9800'
    }
  ];
} 