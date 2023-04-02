import { Component } from '@angular/core';
import { HttpClient } from '@angular/common/http';

@Component({
  selector: 'app-new-project',
  templateUrl: './new-project.component.html',
  styleUrls: ['./new-project.component.css']
})
export class NewProjectComponent {

  constructor(private http: HttpClient) {}

  startTraining() {
    const files = (document.querySelector('input[type="file"]') as HTMLInputElement).files;
    const formData = new FormData();
    for (let i = 0; i < files.length; i++) {
      formData.append('files', files[i], files[i].name);
    }
    this.http.post('/train', formData).subscribe(
      response => console.log(response),
      error => console.error(error)
    );
  }


  

}
