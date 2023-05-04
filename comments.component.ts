import { Component, OnInit} from '@angular/core';
import { AuthService } from '@auth0/auth0-angular';
import { WebService } from './web.service';
import { FormBuilder, Validators } from '@angular/forms';
import { HttpClient } from '@angular/common/http';
import { NgModule } from '@angular/core';
import { ActivatedRoute } from '@angular/router';

@Component({
    selector: 'comments',
    templateUrl: './comments.component.html',
    styleUrls: ['./comments.component.css']
   })

   export class CommentsComponent{
    discussion_list: any=[];
    comments: any;
    commentForm: any;
    updateDiscussionForm: any;
    page:number =1;
    updateCommentForm: any;
    showForm = false;

    constructor(
        public authService : AuthService, 
                public webService: WebService, 
                private formBuilder: FormBuilder,
                private http: HttpClient,
                private route: ActivatedRoute,){}

    ngOnInit(){
        this.postComment();
        this.updateDiscussion();
        this.updateComment();
    }

    postComment(){
        this.commentForm = this.formBuilder.group({
            username: ['', Validators.required],
            comment: ['', Validators.required],
            emotions: 'sad'
        });

        this.discussion_list= this.webService.getDiscussion(this.route.snapshot.params['id']);
        this.comments = this.webService.getComments(this.route.snapshot.params['id']);
    }

    updateDiscussion(){
        this.updateDiscussionForm = this.formBuilder.group({
            title: ['', Validators.required],
            content: ['', Validators.required],
        });

        this.discussion_list = this.webService.getDiscussion(this.route.snapshot.params['id']);
    }

    updateComment(){
        this.updateCommentForm = this.formBuilder.group({
            comment: ['', Validators.required],
            emotions: 'sad'
        });

        this.discussion_list = this.webService.getDiscussion(this.route.snapshot.params['id']);
        this.comments = this.webService.getComments(this.route.snapshot.params['id']);
    }

    onSubmitComment(){
        this.webService.postComment(this.commentForm.value).subscribe((response: any) =>{
            this.commentForm.reset();
            this.comments = this.webService.getComments(this.route.snapshot.params['id'])
        })
    }

    onSubmitDiscussion(){
        this.webService.putDiscussion(this.updateDiscussionForm.value).subscribe((response: any)=> {
            this.updateDiscussionForm.reset();
            this.discussion_list= this.webService.getDiscussion(this.route.snapshot.params['id'])
        })
    }

    isInvalidComment(control:any){
      return this.commentForm.controls[control].invalid && this.commentForm.controls[control].touched;
    }

    isUntouchedComment(){
      return this.commentForm.controls.username.pristine || this.commentForm.controls.comment.pristine;
   }

    isIncompleteComment(){
      return this.isInvalidComment('username') || this.isInvalidComment('comment') || this.isUntouchedComment();
    }


   isIncompleteDiscussion(){
       return this.isInvalidUpdateDiscussion('title') || this.isInvalidUpdateDiscussion('content') || this.isUntouchedUpdateDiscussion();
   }

    isInvalidUpdateDiscussion(control:any){
        return this.updateDiscussionForm.controls[control].invalid && this.updateDiscussionForm.controls[control].touched;
   }

   isUntouchedUpdateDiscussion(){
      return this.updateDiscussionForm.controls.title.pristine || this.updateDiscussionForm.controls.content.pristine;
    }


    displayStyle = "none";

    openFullScreen(){
        this.displayStyle ="block";
    }

    openPopup(){
        this.displayStyle ="block";
    }

    closePopUp(){
        this.displayStyle ="none";
    }

    closeFullScreen(){
        this.displayStyle="none";
    }

    displayForm(){
        this.showForm = true;
    }

    onSubmitDeleteDiscussion(){
        if(confirm("Do you want to delete this discussion"))
        this.webService.deleteDiscussion(this.route.snapshot.params['id'])
        .subscribe(response =>{
            this.discussion_list = this.webService.getDiscussions(this.page);
            window.location.href = "http://localhost:4200/discussion"
        })
    }

    onSubmitDeleteComment(id:any){
        if(confirm("Do you want to delete this comment"))
        this.webService.deleteComment(id).subscribe(response =>{
            this.comments = this.webService.getComments(this.route.snapshot.params['id'])
        })
    }

    }