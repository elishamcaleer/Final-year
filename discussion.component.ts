import { Component, OnInit} from '@angular/core';
import { AuthService } from '@auth0/auth0-angular';
import { WebService } from './web.service';
import { FormBuilder, Validators } from '@angular/forms';
import { HttpClient } from '@angular/common/http';
import { NgModule } from '@angular/core';
import { ActivatedRoute } from '@angular/router';

@Component({
    selector: 'discussion',
    templateUrl: './discussion.component.html',
    styleUrls: ['./discussion.component.css']
   })

   export class DiscussionComponent{ 
    discussion_list: any = [];
    discussionForm : any;
    page : number = 1;
    pagesToShow = 5;
    showForm = false;
    searchResults: any = [];


    constructor(public authService : AuthService, 
                public webService: WebService, 
                private formBuilder: FormBuilder,
                private http: HttpClient,
                private route: ActivatedRoute) {}

    ngOnInit(){
        this.sessionStorage();
        this.newDiscussion();
    }

    displayForm(){
        this.showForm = true;
    }

    sessionStorage(){
        if (sessionStorage['page']) {
            this.page = Number(sessionStorage['page']) 
        }
        this.discussion_list = this.webService.getDiscussions(this.page);
    }

    previousPage(){
        if (this.page > 1){
            this.page = this.page -1;
            sessionStorage['page'] = this.page;
            this.discussion_list = this.webService.getDiscussions(this.page);
        }
       
    }

    nextPage(){
        this.page = this.page + 1;
        sessionStorage['page'] = this.page;
        this.discussion_list = this.webService.getDiscussions(this.page);
    }

    newDiscussion(){
        this.discussionForm = this.formBuilder.group({
            username: ['', Validators.required],
            title: ['', Validators.required],
            content: ['', Validators.required]
        });
    }

    onSubmit(){
        console.log(this.discussionForm.value);
        this.webService.postDiscussion(this.discussionForm.value)
            .subscribe((response: any) => {
                this.discussionForm.reset();
                this.discussion_list = this.webService.getDiscussions(this.page);
                    {window.scrollTo(0,0);}
            }) 
    }

    isInvalid(control:any){
        return this.discussionForm.controls[control].invalid &&
               this.discussionForm.controls[control].touched;
    }

    isUntouched(){
        return this.discussionForm.controls.username.pristine ||
               this.discussionForm.controls.title.pristine ||
               this.discussionForm.controls.content.pristine; 
    }

    isIncomplete() {
        return this.isInvalid('username') ||
               this.isInvalid('title') ||
               this.isInvalid('content') ||
               this.isUntouched();
    }

    displayStyle = "none";

    openFullScreen(){
        this.displayStyle = "block";
    }

    closePopup(){
        this.displayStyle = "none";
    }

    searchDiscussions(query: string) {
        this.searchResults = this.discussion_list.filter(
          (discussion: any) =>
            discussion.username.toLowerCase().includes(query.toLowerCase()) ||
            discussion.title.toLowerCase().includes(query.toLowerCase()) ||
            discussion.content.toLowerCase().includes(query.toLowerCase())
        );
    }


}
