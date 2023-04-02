import { Component } from '@angular/core';
import { Router } from '@angular/router';

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css','./neural.css'],
})

export class AppComponent {
  constructor(private router: Router) {}

  startProject() 
  {
    this.router.navigate(['newproject']);
  }
  title = 'PixelVisionX';
}