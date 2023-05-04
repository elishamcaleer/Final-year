import { Component, OnInit} from '@angular/core';
import { AuthService } from '@auth0/auth0-angular';
import { WebService } from './web.service';
import { FormBuilder, Validators } from '@angular/forms';
import { HttpClient } from '@angular/common/http';
import { NgModule } from '@angular/core';

interface SentimentResponse {
    sentiment: number;
}
@Component({
 selector: 'sentimentanalysis',
 templateUrl: './sentimentanalysis.component.html',
 styleUrls: ['./sentimentanalysis.component.css']
})
export class SentimentAnalysisComponent{ 
    sentence = '';
    sentiment: any;

    constructor(public authService : AuthService, 
                public webService: WebService, 
                private formBuilder: FormBuilder,
                private http: HttpClient,) {}

  

    getSentiment() {
        this.http.post<SentimentResponse>('http://127.0.0.1:5000/api/v1.0/sentimentanalysis', { sentence: this.sentence })
          .subscribe((data) => {
            this.sentiment = data.sentiment;
            console.log(this.sentiment)
          });

      }
}
