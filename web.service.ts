import { HttpClient } from '@angular/common/http';
// import { ThisReceiver } from '@angular/compiler';
import { Observable } from 'rxjs';
import { Injectable } from '@angular/core';
import { AuthService } from '@auth0/auth0-angular';

@Injectable()

export class WebService {

   private discussionID : any;

   private commentID : any;

   private assessmentID : any;


   constructor(private http: HttpClient,
      private authService: AuthService) { }


   getWordcloud(){
      return this.http.get('http://127.0.0.1:5000/api/v1.0')
   }

   analyzeSentiment(text: string): Observable<any> {
      return this.http.post('http://127.0.0.1:5000/api/v1.0/sentimentanalysis', {text});
    }

    getData(): Observable<any> {
      return this.http.get<any>('assets/sentiment_analysis.json');
    }

    getDiscussions(page:number){
      return this.http.get('http://127.0.0.1:5000/api/v1.0/discussion?pn=' + page);
   }

   postDiscussion(discussion_list:any){
      let postData = new FormData();
      postData.append("username", discussion_list.username);
      postData.append("title", discussion_list.title);
      postData.append("content", discussion_list.content);

      return this.http.post('http://127.0.0.1:5000/api/v1.0/discussion', postData);
   }

   getDiscussion(id: any){
      this.discussionID = id;
      return this.http.get('http://127.0.0.1:5000/api/v1.0/discussion/' + id)
   }

   putDiscussion(discussion_list: any){
      let postData = new FormData();
      postData.append("title", discussion_list.title);
      postData.append("content", discussion_list.content);

      return this.http.put('http://127.0.0.1:5000/api/v1.0/discussion/' + this.discussionID, postData);
   }

   deleteDiscussion(id: any){
      this.discussionID = id;
      return this.http.delete('http://127.0.0.1:5000/api/v1.0/discussion/' + this.discussionID)
   }

   getComments(id: any){
      return this.http.get('http://127.0.0.1:5000/api/v1.0/discussion/' + id  + '/comments');
   }

   getComment(comment : any){
      this.commentID = comment;
      return this.http.get('http://127.0.0.1:5000/api/v1.0/discussion/' + this.discussionID + '/comments/' + comment);
   }

   postComment(comment: any){
      let postData = new FormData();
      postData.append("username", comment.username);
      postData.append("comment", comment.comment);
      postData.append("emotions", comment.emotions);

      return this.http.post('http://127.0.0.1:5000/api/v1.0/discussion/' + this.discussionID + '/comments', postData);
   }
   
   putComment(comment: any){
      this.commentID = comment;
      let postData = new FormData();
      postData.append("comment", comment.comment);
      postData.append("emotions", comment.emotions);

      return this.http.put('http://127.0.0.1:5000/api/v1.0/discussion/' + this.discussionID + '/comments/' + this.commentID, postData);
   }

   deleteComment(comment: any){
      this.commentID = comment;
      return this.http.delete('http://127.0.0.1:5000/api/v1.0/discussion/' + this.discussionID + '/comments/' + comment);
   }

   getCount(): Observable<number> {
      return this.http.get<number>('http://127.0.0.1:5000/api/v1.0/count');
    }
    
    getAssessmentCount(): Observable<number> {
      return this.http.get<number>('http://127.0.0.1:5000/api/v1.0/assessmentcount');
    }

   postAssessment(assessment_list:any){
      let postData = new FormData();
      postData.append("question1", assessment_list.question1);
      postData.append("question2", assessment_list.question2);
      postData.append("question3", assessment_list.question3);
      postData.append("question4", assessment_list.question4);
      postData.append("question5", assessment_list.question5);
      postData.append("question6", assessment_list.question6);
      postData.append("question6_select", assessment_list.question6_select);

      return this.http.post('http://127.0.0.1:5000/api/v1.0/self-assess', postData);
   }
   
   getAssessment(id:any){
      this.assessmentID = id;
      return this.http.get('http://127.0.0.1:5000/api/v1.0/self-assess/' + id)

   }

   getModelData(): Observable<any> {
      return this.http.get<any>('assets/model_output.json');
    }

}
