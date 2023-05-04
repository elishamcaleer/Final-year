import { Component } from '@angular/core';
import { AuthService } from '@auth0/auth0-angular';

@Component({
 selector: 'profile',
 templateUrl: './profile.component.html',
 styleUrls: ['./profile.component.css']
})
export class ProfileComponent { 
    constructor(public authService : AuthService) {}
}