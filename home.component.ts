import { HttpClient } from '@angular/common/http';
import { Component } from '@angular/core';
import { ActivatedRoute } from '@angular/router';
import { AuthService } from '@auth0/auth0-angular';
import { WebService } from './web.service';
import { Observable, timer } from 'rxjs';
import { switchMap } from 'rxjs/operators';


@Component({
 selector: 'home',
 templateUrl: './home.component.html',
 styleUrls: ['./home.component.css']
})
export class HomeComponent{
    count: number = 0;
    assesscount: number = 0;
    constructor(public authService : AuthService,
                private route: ActivatedRoute,
                private http: HttpClient,
                public webService: WebService, ) {}


  ngOnInit() {
    this.webService.getCount().subscribe(count => {
      this.count = count;
    });
    this.webService.getAssessmentCount().subscribe(assesscount => {
      this.assesscount = assesscount;
    });
  }
}

