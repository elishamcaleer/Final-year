import { Component, OnInit} from '@angular/core';
import { AuthService } from '@auth0/auth0-angular';
import { WebService } from './web.service';
import { FormBuilder, Validators } from '@angular/forms';
import { HttpClient } from '@angular/common/http';
import { NgModule } from '@angular/core';
import { ActivatedRoute } from '@angular/router';

@Component({
    selector: 'visualisation',
    templateUrl: './visualisation.component.html',
    styleUrls: ['./visualisation.component.css']
   })

   export class VisualisationComponent{ 
    data : any;


    constructor(public authService : AuthService, 
                public webService: WebService, 
                private formBuilder: FormBuilder,
                private http: HttpClient,
                private route: ActivatedRoute) {}

    displayStyle = "none";

    openFullScreen() {
        this.displayStyle = "block";
        this.webService.getModelData().subscribe(data => {
            this.data = data;
        });
    }
            
    closePopup() {
        this.displayStyle = "none";
    }

}