import { Component } from '@angular/core';

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css','./neural.css'],
})
export class AppComponent {
  startProject() {
    window.location.href = 'newproject.html';
  }
  title = 'PixelVisionX';
}