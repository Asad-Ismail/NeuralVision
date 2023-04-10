import { HttpClient } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { Observable } from 'rxjs';

@Injectable({
  providedIn: 'root'
})
export class NewProjectService {
  private apiUrl = 'http://localhost:5000/api/newproject';

  constructor(private http: HttpClient) { }

  createNewProject(name: string): Observable<any> {
    return this.http.post<any>(this.apiUrl, { name });
  }
}
