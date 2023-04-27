import { Component, OnInit, OnDestroy, NgZone } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { ChartDataset, ChartOptions } from 'chart.js';
import { FormGroup, FormControl, Validators } from '@angular/forms';
import io from 'socket.io-client';
import { Router } from '@angular/router';


@Component({
  selector: 'app-instance-segmentation',
  templateUrl: './instance-segmentation.component.html',
  styleUrls: ['./instance-segmentation.component.css']
})


export class InstanceSegmentationComponent implements OnInit, OnDestroy{

  logs: string = '';
  metrics: any[] = [];
  trainingStarted: boolean = false;
  // Add a new property to store hyperparameters form
  hyperparametersForm: FormGroup;
  // Add a new property to check if hyperparameters have been submitted
  hyperparametersSubmitted: boolean = false;
  private socket: any;
  dataPath: string = '';
  trainImagesCount: number | null = null;
  trainlabelsCount: number | null = null;
  valImagesCount: number | null = null;
  vallabelsCount: number | null = null;


  onDataPathChange(value: string) {
    this.dataPath = value;
  }
  

  async sendDataToBackend() {
    if (this.dataPath) {
      const requestData = {
        dataPath: this.dataPath,
      };
  
      try {
        const response: any = await this.http.post('http://localhost:5000/api/instance_segmentation_uploaddata', requestData).toPromise();
        console.log(response);
        // Get the number of train images and labels from the response
        const trainImagesCount = response.trainImagesCount;
        const labelsCount = response.labelsCount;
        const valImagesCount = response.valImagesCount;
        const vallabelsCount = response.vallabelsCount;
        // Display the counts on the front end or use them as needed
      } catch (error) {
        console.error('Sending data to backend failed', error);
        // Handle the failure (retry or inform the user)
      }
    }
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
  uploadInProgress = false;
  uploadProgress = 0;
  showSteps = false;

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
