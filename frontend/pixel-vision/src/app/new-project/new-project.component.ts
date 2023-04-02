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
  
  files: File[] = [];

  onFileSelected(event: any) {
    this.files = event.target.files;
  }

  startTraining() {
    const formData = new FormData();
    if (this.files && this.files.length > 0) {
      for (let i = 0; i < this.files.length; i++) {
        formData.append('files', this.files[i], this.files[i].name);
      }
      this.http.post('/train', formData).subscribe(
        response => console.log(response),
        error => console.error(error)
      );
    }
  }

}  