import { HttpClient } from '@angular/common/http';
// import { ThisReceiver } from '@angular/compiler';
import { Observable } from 'rxjs';
import { Injectable } from '@angular/core';
@Injectable()

export class WebService {

   private discussionID : any;

   private commentID : any;

   constructor(private http: HttpClient) { }


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
}
