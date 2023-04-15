import { Component } from '@angular/core';
import { Router } from '@angular/router';

@Component({
  selector: 'app-task-selection',
  templateUrl: './task-selection.component.html',
  styleUrls: ['./task-selection.component.css'],
})
export class TaskSelectionComponent {
  tasks = [
    { name: 'Classification', route: '/classification' },
    { name: 'Segmentation', route: '/segmentation' },
    { name: 'Object Detection', route: '/object-detection' },
    { name: 'Instance Segmentation', route: '/instance-segmentation' },
  ];

  constructor(private router: Router) {}

  onTaskSelect(task: any) {
    console.log('Selected task:', task.name);
    this.router.navigate([task.route]);
  }
}
