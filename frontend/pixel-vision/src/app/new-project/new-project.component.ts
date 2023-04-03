import { Component } from '@angular/core';
import { HttpClient } from '@angular/common/http';

@Component({
  selector: 'app-new-project',
  templateUrl: './new-project.component.html',
  styleUrls: ['./new-project.component.css']
})


export class NewProjectComponent 

{

  constructor(private http: HttpClient) {}

  startTraining() {
    this.http.get('http://localhost:5000/api/start_training').subscribe(data => {
      console.log(data);
    });
  }

}  