import { Component, OnInit} from '@angular/core';
import { AuthService } from '@auth0/auth0-angular';
import { WebService } from './web.service';
import { FormBuilder, FormControl, Validators } from '@angular/forms';
import { HttpClient } from '@angular/common/http';
import { NgModule } from '@angular/core';
import { ActivatedRoute } from '@angular/router';
import { Observable } from 'rxjs';

@Component({
    selector: 'self-assess',
    templateUrl: './self-assess.component.html',
    styleUrls: ['./self-assess.component.css']
   })

   export class AssessComponent{ 
    page : number = 1;
    assessment_form: any;
    assessment_list: any;
    question3 = new FormControl([]);
    maxSelections = 3;
    assessmentID: any;

    constructor(public authService : AuthService, 
        public webService: WebService, 
        private formBuilder: FormBuilder,
        private http: HttpClient,
        private route: ActivatedRoute) {}

        ngOnInit(){
            this.newAssessment();
        }

       

        onSubmit(){
            confirm("Reflection on your day is useful")
            console.log(this.assessment_form.value);
            this.webService.postAssessment(this.assessment_form.value)
                .subscribe((response: any) => {
                    this.assessment_form.reset();
                        {window.scrollTo(0,0);}
                }) 
        }

        newAssessment(){
            this.assessment_form = this.formBuilder.group({
                question1: ['', Validators.required],
                question2: ['', Validators.required],
                question3: [[], [Validators.required, , this.validateEmotions(3)]],
                question4: ['', Validators.required],
                question5: ['', Validators.required],
                question6: ['', Validators.required], 
                question6_select: ['']
            });
        }

        isInvalid(control:any){
            return this.assessment_form.controls[control].invalid &&
                   this.assessment_form.controls[control].touched;
        }


    isUntouched(){
        return this.assessment_form.controls.question1.pristine ||
               this.assessment_form.controls.question2.pristine ||
               this.assessment_form.controls.question3.pristine ||
               this.assessment_form.controls.question4.pristine ||
               this.assessment_form.controls.question5.pristine ||
               this.assessment_form.controls.question6.pristine; 
    }

    isIncomplete() {
        return this.isInvalid('question1') ||
               this.isInvalid('question2') ||
               this.isInvalid('question3') ||
               this.isInvalid('question4') ||
               this.isInvalid('question5') ||
               this.isInvalid('question6') ||
               this.isUntouched();
    }

    validateEmotions(maxSelections: number) {
        return (control: FormControl) => {
            let selectedEmotions = control.value;
            if (selectedEmotions && selectedEmotions.length > maxSelections) {
                return {
                    'invalidEmotions': true
                };
            }
            return null;
        };
    }

    redirectToOutput(id:any){
        this.assessmentID = id;
        return this.http.get('http://127.0.0.1:5000/api/v1.0/self-assess/' + id)
    }

}