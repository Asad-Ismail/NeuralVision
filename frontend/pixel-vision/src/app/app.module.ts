import { NgModule,CUSTOM_ELEMENTS_SCHEMA } from '@angular/core';
import { RouterModule, Routes } from '@angular/router';
import { BrowserModule } from '@angular/platform-browser';
import { AppComponent } from './app.component';
import { StartPageComponent } from './start-page/start-page.component';
import { NewProjectComponent } from './new-project/new-project.component';
import { HttpClientModule } from '@angular/common/http';
import { NgChartsModule } from 'ng2-charts';
import { ReactiveFormsModule } from '@angular/forms';
import { NewProjectFormComponent } from './new-project/new-project-form/new-project-form.component';
import { TaskSelectionComponent } from './task-selection/task-selection.component';
import { InstanceSegmentationComponent } from './instance-segmentation/instance-segmentation.component';


const routes: Routes = [
  { path: '', component: StartPageComponent },
  { path: 'new-project-start', component: NewProjectFormComponent},
  { path: 'new-project', component: NewProjectComponent},
  { path: 'task-selection', component: TaskSelectionComponent},
  { path: 'instance-segmentation', component: InstanceSegmentationComponent},
];

@NgModule({
  declarations: [
    AppComponent,
    NewProjectComponent,
    StartPageComponent,
    NewProjectFormComponent,
    TaskSelectionComponent,
    InstanceSegmentationComponent
  ],
  imports: [
    BrowserModule,
    HttpClientModule,
    NgChartsModule,
    ReactiveFormsModule,
    RouterModule.forRoot(routes)
  ],
  providers: [],
  bootstrap: [AppComponent],
  schemas: [CUSTOM_ELEMENTS_SCHEMA]
})
export class AppModule { }
