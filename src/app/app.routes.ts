import { Routes } from '@angular/router';

export const routes: Routes = [
  {
    path: '',
    redirectTo: '/home',
    pathMatch: 'full'
  },
  {
    path: 'home',
    loadComponent: () => import('./components/home/home.component').then(m => m.HomeComponent)
  },
  {
    path: 'maze-solver',
    loadComponent: () => import('./components/maze-solver/maze-solver.component').then(m => m.MazeSolverComponent)
  },
  {
    path: 'ai-trainer',
    loadComponent: () => import('./components/ai-trainer/ai-trainer.component').then(m => m.AiTrainerComponent)
  },
  {
    path: 'neural-network',
    loadComponent: () => import('./components/neural-network-viz/neural-network-viz.component').then(m => m.NeuralNetworkVizComponent)
  },
  {
    path: '**',
    redirectTo: '/home'
  }
];
