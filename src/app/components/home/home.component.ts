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
      description: 'Train reinforcement learning agents using Q-Learning and Deep Q-Networks (DQN) with real-time neural network visualization',
      icon: 'school',
      route: '/ai-trainer',
      color: '#2196F3'
    }
  ];
}