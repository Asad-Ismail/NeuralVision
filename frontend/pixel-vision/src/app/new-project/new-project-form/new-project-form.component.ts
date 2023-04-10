import { Component } from '@angular/core';
import { FormControl, FormGroup, Validators } from '@angular/forms';
import { Router } from '@angular/router';
import { NewProjectService } from '../new-project.service';

@Component({
  selector: 'app-new-project-form',
  templateUrl: './new-project-form.component.html',
  styleUrls: ['./new-project-form.component.css']
})
export class NewProjectFormComponent {
  projectForm = new FormGroup({
    name: new FormControl('', Validators.required),
  });

  constructor(private newProjectService: NewProjectService, private router: Router) {}

  onSubmit() {
    if (typeof this.projectForm.value.name === 'string') {
      this.newProjectService.createNewProject(this.projectForm.value.name).subscribe(() => {
        //this.router.navigate(['/new-project', this.projectForm.value.name]);
        this.router.navigate(['/new-project']);
      });
    } else {
      // Handle the case when the name is not a string (e.g., show an error message)
    }
  }
}
