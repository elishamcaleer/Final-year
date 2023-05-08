import { BrowserModule } from '@angular/platform-browser';
import { NgModule } from '@angular/core';
import { AppComponent } from './app.component';
import { RouterModule } from '@angular/router';
import { WebService } from './web.service';
import { HttpClientModule } from '@angular/common/http';
import { HomeComponent } from './home.component';
import { ReactiveFormsModule } from '@angular/forms';
import { AuthModule } from '@auth0/auth0-angular';
import { NavComponent } from './nav.component';
import { ProfileComponent } from './profile.component';
import { AnalysisComponent } from './analysis.component';
import { SentimentAnalysisComponent } from './sentimentanalysis.component';
import { DiscussionComponent } from './discussion.component';
import { CommentsComponent } from './comments.component';
import { FormsModule } from '@angular/forms';
import { AssessComponent } from './self-assess.component';
import { VisualisationComponent } from './visualisation.component';

var routes: any = [
    {
        path: '',
        component: HomeComponent
    },
    {
        path: 'profile',
        component: ProfileComponent
    },
    {
        path: 'analysis',
        component: AnalysisComponent
    },
    {
        path: 'sentimentanalysis',
        component: SentimentAnalysisComponent
    },
    {
        path: 'discussion',
        component: DiscussionComponent
    },
    {
        path: 'discussion/:id',
        component: CommentsComponent
    },
    {
        path: 'self-assess',
        component: AssessComponent
    },
    {
        path: 'visualisation',
        component: VisualisationComponent
    }

];


@NgModule({
    declarations: [
        AppComponent, HomeComponent, NavComponent, ProfileComponent, AnalysisComponent, SentimentAnalysisComponent, DiscussionComponent, CommentsComponent, AssessComponent, VisualisationComponent
    ],
    imports: [
        BrowserModule, HttpClientModule, RouterModule.forRoot(routes), ReactiveFormsModule, FormsModule,
        AuthModule.forRoot({
            domain: 'dev-md5wvsh1lbkeegp8.us.auth0.com',
            clientId: 'tngS58o32LsxgL2lx2ZGlukXwfe9ujYZ'
        })
    ],
    providers: [WebService],
    bootstrap: [AppComponent]
})
export class AppModule { }