import { NgModule } from '@angular/core';
import { RouterModule, Routes } from '@angular/router';
import { BrowserModule } from '@angular/platform-browser';
import { AppComponent } from './app.component';
import { StartPageComponent } from './start-page/start-page.component';
import { NewProjectComponent } from './new-project/new-project.component';
import { HttpClientModule } from '@angular/common/http';
import { ChartsModule } from 'ng2-charts';


const routes: Routes = [
  { path: '', component: StartPageComponent },
  { path: 'new-project', component: NewProjectComponent }
];

@NgModule({
  declarations: [
    AppComponent,
    NewProjectComponent,
    StartPageComponent
  ],
  imports: [
    BrowserModule,
    HttpClientModule,
    ChartsModule,
    RouterModule.forRoot(routes)
  ],
  providers: [],
  bootstrap: [AppComponent]
})
export class AppModule { }
