import { Component } from '@angular/core';
import { AuthService } from '@auth0/auth0-angular';
import { WebService } from './web.service';


@Component({
 selector: 'analysis',
 templateUrl: './analysis.component.html',
 styleUrls: ['./analysis.component.css']
})
export class AnalysisComponent { 
    
    data: any;

    constructor(public authService : AuthService,
                public webService: WebService ) {}

    displayStyle = "none";

    openFullScreen() {
           this.displayStyle = "block";
            this.webService.getData().subscribe(data => {
              this.data = data;
            });
    }

    closePopup() {
        this.displayStyle = "none";
    }

   
}
