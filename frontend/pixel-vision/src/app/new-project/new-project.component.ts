import { Component, OnInit, OnDestroy, NgZone } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { ChartDataset, ChartOptions } from 'chart.js';
import { FormGroup, FormControl, Validators } from '@angular/forms';
import io from 'socket.io-client';
import { Router } from '@angular/router';
import { HostListener } from '@angular/core';



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

  imagePreviews: string[] = [];
  imagesUploaded = false;

  async submitDataPath() {
    await this.sendDataToBackend();
  }

  @HostListener('window:beforeunload', ['$event'])
  unloadHandler(event: Event) {
    this.ngOnDestroy();
    event.returnValue = true;
  }

  @HostListener('window:unload', ['$event'])
  unloadHandlerOnClose(event: Event) {
    this.ngOnDestroy();
    event.returnValue = true;
  }

  reset() {
    this.logs = '';
    this.metrics = [];
    this.trainingStarted = false;
    this.lineChartData[0].data = [];
    this.lineChartLabels = [];
    this.trainingStatus = '';
    this.hyperparametersSubmitted = false; // Add this property to your class and set it to true after submitting the form
    // Reset the hyperparameters form with default values
    this.hyperparametersForm.reset({
      learningRate: 0.001,
      epochs: 10,
      batchSize: 32,
      algorithmType: 'MOCO'
    });
    // stop the training
    this.http.get('http://localhost:5000/api/stop_training').subscribe(
      (response) => {
        console.log(response);
      },
      (error) => {
        console.log(error);
      }
    );  
  }  

  stopTraining() {
    const headers = new Headers({ 'Content-Type': 'application/json' });
    const blob = new Blob([JSON.stringify({ stopTraining: true })], { type: 'application/json' });
    navigator.sendBeacon('http://localhost:5000/api/stop_training', blob);
  }
  

  public lineChartData: ChartDataset[] = [
    {
      data: [],
      label: 'Training Loss',
      borderColor: 'blue',
      backgroundColor: 'rgba(0, 0, 255, 0.1)',
    },
    {
      data: [],
      label: 'Validation Loss',
      borderColor: 'red',
      backgroundColor: 'rgba(255, 0, 0, 0.1)',
    },
  ];
  
  public lineChartLabels: string[] = [];
  public lineChartOptions: ChartOptions = { responsive: true };
  public lineChartLegend = true;
  public lineChartType: 'line' = 'line';
  public lineChartPlugins = [];
  
  public trainingStatus: string = '';
  
  showSteps = false;
  dataPath: string = '';
  trainImagesCount: number | null = null;


  onDataPathChange(value: string) {
    this.dataPath = value;
  }
  

  async sendDataToBackend() {
    if (this.dataPath) {
      const requestData = {
        dataPath: this.dataPath,
      };
  
      try {
        const response: any = await this.http.post('http://localhost:5000/api/ssl_uploaddata', requestData).toPromise();
        console.log(response);
        // Get the number of train images and labels from the response
        this.trainImagesCount = response.trainImagesCount; // <- Set the property
        // Display the counts on the front end or use them as needed
      } catch (error) {
        console.error('Sending data to backend failed', error);
        // Handle the failure (retry or inform the user)
      }
    }
  }

  constructor(private http: HttpClient, private ngZone: NgZone,private router: Router) 
  
  {
    this.socket = io('http://localhost:5000');
    // Initialize the hyperparameters form with default values
    this.hyperparametersForm = new FormGroup({
      learningRate: new FormControl(0.001, Validators.required),
      // Add more form controls if needed
      epochs: new FormControl(50, Validators.required),
      batchSize: new FormControl(32, Validators.required),
      algorithmType: new FormControl('MOCO', Validators.required)
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

    // stop training if running intially
    this.stopTraining();

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
    //reset everything
    this.reset();
  }

  startTraining() {
    this.trainingStarted = true;
    this.http.get('http://localhost:5000/api/start_training').subscribe((data: any) => {
      //console.log(data.message);
    });
  }

  skipProcess() {
    //  logic for skipping the self-supervised model training process
    this.router.navigate(['task-selection/']);
  }

  continueProcess() {
    //  logic for skipping the self-supervised model training process
    this.showSteps = true;
  }
  

  updateChartData(metric: any) {
    // Assuming your metric object has properties called "train_loss", "val_loss", and "epoch"
    console.log(metric);
    (this.lineChartData[0].data as number[]).push(metric.train_loss);
    (this.lineChartData[1].data as number[]).push(metric.val_loss);
    this.lineChartLabels.push(metric.epoch.toString());
    // Use the ngZone service to update the chart in the Angular zone
    this.ngZone.run(() => {
      this.lineChartData = [...this.lineChartData];
      this.lineChartLabels = [...this.lineChartLabels];
    });
  }
  

}
