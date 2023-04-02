import { NgModule } from '@angular/core';
import { RouterModule, Routes } from '@angular/router';
import { BrowserModule } from '@angular/platform-browser';
import { AppComponent } from './app.component';
import { NewProjectComponent } from './new-project/new-project.component';


const routes: Routes = [
  { path: '', component: AppComponent }, // set the default route to AppComponent
  { path: 'new-project', component: NewProjectComponent }
];

@NgModule({
  declarations: [
    AppComponent,
    NewProjectComponent
  ],
  imports: [
    BrowserModule,
    RouterModule.forRoot(routes)
  ],
  exports: [RouterModule],
  providers: [],
  bootstrap: [AppComponent]
})

export class AppModule { }
export class AppRoutingModule { }
