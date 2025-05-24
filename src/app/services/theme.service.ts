import { Injectable } from '@angular/core';
import { BehaviorSubject } from 'rxjs';

@Injectable({
  providedIn: 'root'
})
export class ThemeService {
  private readonly THEME_KEY = 'maze-runner-theme';
  private readonly isDarkThemeSubject = new BehaviorSubject<boolean>(this.getInitialTheme());

  isDarkTheme$ = this.isDarkThemeSubject.asObservable();

  constructor() {
    this.applyTheme(this.isDarkThemeSubject.value);
  }

  private getInitialTheme(): boolean {
    // Check localStorage first
    const savedTheme = localStorage.getItem(this.THEME_KEY);
    if (savedTheme !== null) {
      return savedTheme === 'dark';
    }
    
    // Fall back to system preference
    return window.matchMedia('(prefers-color-scheme: dark)').matches;
  }

  toggleTheme(): void {
    const newTheme = !this.isDarkThemeSubject.value;
    this.setTheme(newTheme);
  }

  setTheme(isDark: boolean): void {
    this.isDarkThemeSubject.next(isDark);
    this.applyTheme(isDark);
    localStorage.setItem(this.THEME_KEY, isDark ? 'dark' : 'light');
  }

  private applyTheme(isDark: boolean): void {
    const body = document.body;
    if (isDark) {
      body.classList.add('dark-theme');
    } else {
      body.classList.remove('dark-theme');
    }
  }

  getCurrentTheme(): boolean {
    return this.isDarkThemeSubject.value;
  }
}