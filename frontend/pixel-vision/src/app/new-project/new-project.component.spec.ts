import { Component, OnInit } from '@angular/core';

@Component({
  selector: 'app-new-project',
  templateUrl: './new-project.component.html',
  styleUrls: ['./new-project.component.css']
})
export class NewProjectComponent implements OnInit {

  constructor() { }

  ngOnInit(): void {
  }

  onFileSelected(event: any) {
    // handle file selection logic here
  }

  startTraining() {
    // get selected files
    const files = (document.querySelector('input[type=file]') as HTMLInputElement).files;
    if (!files || files.length === 0) {
      // no files selected, display error message to user
      alert('Please select at least one file to start training.');
      return;
    }

    // create a FormData object to upload the files to the server
    const formData = new FormData();
    for (let i = 0; i < files.length; i++) {
      formData.append('files', files[i]);
    }

    // send the files to the server and start the training process
    fetch('/api/train', {
      method: 'POST',
      body: formData
    }).then(response => {
      if (!response.ok) {
        throw new Error('Network response was not ok');
      }
      return response.json();
    }).then(data => {
      console.log(data);
      // display success message to user and navigate to results page
      alert('Training completed successfully!');
      window.location.href = '/results';
    }).catch(error => {
      console.error('Error during training:', error);
      // display error message to user
      alert('Error during training. Please try again later.');
    });
  }
}
