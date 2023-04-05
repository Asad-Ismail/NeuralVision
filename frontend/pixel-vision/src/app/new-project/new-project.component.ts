import { Component, OnInit, OnDestroy, NgZone } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { ChartDataset, ChartOptions } from 'chart.js';
import { FormGroup, FormControl, Validators } from '@angular/forms';
import io from 'socket.io-client';

@Component({
  selector: 'app-new-project',
  templateUrl: './new-project.component.html',
  styleUrls: ['./new-project.component.css']
})

export class NewProjectComponent implements OnInit, OnDestroy 
{

  logs: string = '';
  metrics: any[] = [];
  trainingStarted: boolean = false;
  // Add a new property to store hyperparameters form
  hyperparametersForm: FormGroup;
  // Add a new property to check if hyperparameters have been submitted
  hyperparametersSubmitted: boolean = false;
  private socket: any;


  reset() {
    this.logs = '';
    this.metrics = [];
    this.trainingStarted = false;
    this.lineChartData[0].data = [];
    this.lineChartLabels = [];
    this.trainingStatus = '';
    this.hyperparametersSubmitted = false; // Add this property to your class and set it to true after submitting the form
  }  

  public lineChartData: ChartDataset[] = [
    {
      data: [],
      label: 'Training Loss',
      borderColor: 'black',
      backgroundColor: 'rgba(255,0,0,0.3)',
    },
  ];

  public lineChartLabels: string[] = [];
  public lineChartOptions: ChartOptions = { responsive: true};
  public lineChartLegend = true;
  public lineChartType: 'line' = 'line';
  public lineChartPlugins = [];
  public trainingStatus: string = '';
  

  constructor(private http: HttpClient, private ngZone: NgZone) {

    this.socket = io('http://localhost:5000');
    // Initialize the hyperparameters form with default values
    this.hyperparametersForm = new FormGroup({
      learningRate: new FormControl(0.001, Validators.required),
      // Add more form controls if needed
    });
  }

    // Add a new method to submit hyperparameters
    submitHyperparameters() {
      if (this.hyperparametersForm.valid) {
        this.hyperparametersSubmitted = true;
        const hyperparameters = this.hyperparametersForm.value;
  
        // Send hyperparameters to Flask backend
        this.http.post('http://localhost:5000/api/set_hyperparameters', hyperparameters).subscribe((data: any) => {
          console.log('Hyperparameters submitted:', data.message);
        });
      }
    }

  ngOnInit(): void {
    this.socket.on('log', (log: string) => {
      // Ignore logs for now
    });

    this.socket.on('metric', (metric: any) => {
      this.metrics.push(metric);
      this.updateChartData(metric);
    });

    // Periodically update the status
    setInterval(() => {
      this.http.get('http://localhost:5000/api/status').subscribe((data: any) => {
      console.log("Test log ", data)  
      this.trainingStatus = data.status;
      });
    }, 500); // Update every .25 seconds


  }

  ngOnDestroy(): void {
    this.socket.disconnect();
    this.reset();
    // Send a request to the Flask app to stop the training and reset the variables
  this.http.get('http://localhost:5000/api/reset').subscribe((data: any) => {
    // Handle the response if needed
  });
  }

  startTraining() {
    this.trainingStarted = true;
    this.http.get('http://localhost:5000/api/start_training').subscribe((data: any) => {
      //console.log(data.message);
    });
  }

  updateChartData(metric: any) {
    // Assuming your metric object has a property called "loss" and another called "epoch"
    console.log(metric);
    (this.lineChartData[0].data as number[]).push(metric.loss);
    this.lineChartLabels.push(metric.epoch.toString());
    // Use the ngZone service to update the chart in the Angular zone
    this.ngZone.run(() => {
      this.lineChartData = [...this.lineChartData];
      this.lineChartLabels = [...this.lineChartLabels];
    });
  }
}
