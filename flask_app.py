from flask import Flask, request, jsonify, make_response, render_template, session
from pymongo import MongoClient
from bson import ObjectId
from functools import wraps
from flask_cors import CORS
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from wordcloud import STOPWORDS
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from textblob import TextBlob
import seaborn as sns 
from pandas import Series
from pylab import*
from pandas import DataFrame
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from textblob import TextBlob
import jwt
from typing import List
import bson
import gensim
import pyLDAvis.gensim_models as gensimvis
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.corpora import Dictionary
from nltk.stem.porter import PorterStemmer
import pyLDAvis
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

stemmer = PorterStemmer()

app = Flask(__name__)
CORS(app)

app.config['SECRET_KEY'] = 'mysecret'

client = MongoClient("mongodb://127.0.0.1:27017")
db = client.MentalHealth
Tweets = db.Tweets
discussion_forume = db.forum
users = db.users
self_assessment = db.self_assess

@app.route('/api/v1.0/analysis', methods=['POST', 'GET'])
def wordcloud():

    data = list(db.Tweets.find({}, {'post_text': 1}))
    df = pd.DataFrame(data)
    sw = set(STOPWORDS)
    text = " ".join(i for i in data)
    sw.update(['t', 'co', 'https', 'Im', "RT"])
    
    word_cloud = WordCloud(collocations = False, stopwords = sw, colormap='Blues',mode = "RGBA", background_color=None).generate(text)
    plt.imshow(word_cloud, interpolation='bilinear')
    plt.axis("off")
    plt.savefig('src/assets/wordcloud.png')
    return plt.savefig('src/assets/wordcloud.png', mimetype='image/png')

def cleaning_dataset(data):
    data = Tweets.find()
    df = pd.DataFrame(data)
    df.drop("Unnamed: 0", axis=1, inplace=True)
    stopwords = stopwords.words('english')
    stopwords.extend(['rt', 'amp', "im", "misslusyd", "get", "u", "would"])
    df['post_text']= df['post_text'].apply(lambda x: ' '.join([word for word in x.split() if word.lower() not in stopwords]))
    df['post_text']= df['post_text'].str.replace('\d+', '')
    df['post_text'] = df['post_text'].str.replace('http[^\s]*',"")
    df['post_text'] = df['post_text'].str.replace('[^\w\s]','')
    df['post_text'] = df['post_text'].apply(str.lower)

    return render_template('analysis.component.html', wordcloud=wordcloud, sentiment=sentiment, tweets_per_day_plot=tweets_per_day_plot, tweets_per_hour_plot=tweets_per_hour_plot, tweets_per_year_plot=tweets_per_year_plot, 
    tweets_per_month_plot=tweets_per_month_plot, piechart_top_words=piechart_top_words)

def remove_usernames(tweet):
    tweet = re.sub('@[\w]+','', tweet)
    return tweet


def tokenisation(word):
        word = nltk.word_tokenize(word)
        return word
data = Tweets.find()
df = pd.DataFrame(data)
word = df['post_text'].apply(tokenisation)
df['tokens'] = word

def sentiment():
    data = Tweets.find()
    df = pd.DataFrame(data).apply(cleaning_dataset)
    df = df['post_text'].apply(remove_usernames)
    df = df['post_text'].apply(tokenisation)
    df["Polarity"] = None
    df["Sentiment"] = None
    
    for x, row in df.iterrows():
        tweets = TextBlob(row["post_text"])
        
        polarity = tweets.sentiment.polarity
        sentiment = "Positive" if polarity > 0 else "Negative" if polarity < 0 else "Neutral"
    
    df.at[x, "Polarity"] = polarity
    df.at[x, "Sentiment"] = sentiment

    sentiment_count = df["Sentiment"].value_counts()

    df_json = df[['post_created', 'post_text','Sentiment', 'Polarity']]
    
    df_json.to_json("src/assets/sentiment_analysis.json", orient="records")
    
    sns.set_style("dark")
    sentiment_count.plot(kind="bar", rot=0, colormap='Blues_r', figsize=[26,9])
    plt.xlabel('Sentiment', fontweight= 'bold')
    plt.ylabel('Occurences', fontweight= 'bold')
    plt.title('Sentiment of Tweets', fontsize=20)
    return plt.savefig('src/assets/sentiment.png')


def dataframe(x):
     return Series(dict(Total_Tweets = x['post_text'].count(), 
                        ))

def tweets_per_day_plot(text):
    tweets_by_day = df.groupby(df.index.weekday).apply(dataframe)
    weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    tweets_by_day['day'] = weekdays
    day_of_week_plot = tweets_by_day['Total_Tweets'].plot(kind='bar', figsize=[26,9], colormap='Blues_r')
    plt.xticks(np.arange(7), tweets_by_day.day)
    plt.xlabel('Days of the Week', fontweight= 'bold')
    plt.ylabel('Total Number of Tweets', fontweight= 'bold')
    plt.title('Occurence of Tweets against Days of the Week', fontsize=20)
    plt.imshow(day_of_week_plot)
    for i in range(len(weekdays)):
        text(i, tweets_by_day['Total_Tweets'][i], tweets_by_day['Total_Tweets'][i], ha = 'center')
    return plt.savefig('src/assets/daily.png')

def tweets_per_hour_plot():
    tweets_by_hour = df.groupby(df.index.hour).apply(dataframe)
    tweets_by_hour_plot = tweets_by_hour['Total_Tweets'].plot(kind='line', figsize=[26,9], colormap='Blues_r')
    plt.xlabel('Hours')
    plt.ylabel('Total Number of Tweets')
    plt.title('Occurence of Tweets against Hours of the Days', fontsize=20)
    plt.imshow(tweets_by_hour_plot)
    return plt.savefig('src/assets/hourly.png')

def tweets_per_month_plot(text):
    tweets_by_month = df.groupby(df.index.month).apply(dataframe)
    tweets_by_month_plot = tweets_by_month['Total_Tweets'].plot(kind='bar', figsize=[26,9], colormap='Blues_r')
    plt.xlabel('Months')
    plt.ylabel('Total Number of Tweets')
    plt.title('Occurence of Tweets against Months', fontsize=20)
    months= ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
    tweets_by_month['month'] = months
    plt.xticks(np.arange(12), tweets_by_month.month)
    for i in range(len(months)):
        text(i, tweets_by_month['Total_Tweets'][i+1],tweets_by_month['Total_Tweets'][i+1], ha = 'center')
        
    plt.imshow(tweets_by_month_plot)
    return plt.savefig('src/assets/monthly.png')

def tweets_per_year_plot():
    tweets_per_year = df.groupby(df.index.year).apply(dataframe)
    tweets_per_year_plot = tweets_per_year['Total_Tweets'].plot(kind='line', figsize=[26,9], colormap='Blues_r')
    plt.xlabel('Year')
    plt.ylabel('Total Number of Tweets')
    plt.title('Occurence of Tweets against Years', fontsize=20)
    plt.imshow(tweets_per_year_plot)
    return plt.savefig('src/assets/yearly.png')

def piechart_top_words():
    top_words = df["post_text"].apply(lambda x: ' '.join([word for word in x.split() if word.lower() not in stopwords]))
    top_words = ' '.join(df["post_text"].tolist()).split()
    top_words = [word.lower() for word in top_words]
    topwords_df = DataFrame(Series(top_words).value_counts())  
    topwords_df.columns = ['Frequency']
    piechart = topwords_df['Frequency'][:15].plot(kind='pie', figsize=[26,9], autopct='%.1f%%', colormap='Blues',  wedgeprops={'linewidth': 3.0, 'edgecolor': 'black'})
    plt.title('Pie Chart displaying Top 15 words in Tweets', fontsize=20, )
    plt.imshow(piechart)
    return plt.savefig('src/assets/piechart.png')

@app.route('/api/v1.0/sentimentanalysis', methods=['POST'])
def sentimentpage():
    sentence = request.json['sentence']
    sentiment = TextBlob(sentence).sentiment.polarity
    return jsonify({'sentiment': sentiment})

def sentiment_accuracy():
     df = pd.DataFrame(data).apply(cleaning_dataset)
     df = df['post_text'].apply(remove_usernames)
     df = df['post_text'].apply(tokenisation)
     df_json = df[['post_created', 'post_text','Sentiment', 'Polarity']]
     X_train, X_test, y_train, y_test = train_test_split(df_json['post_text'], df_json['Sentiment'], test_size=0.2, random_state=42)
     vectorizer = CountVectorizer()
     X_train = vectorizer.fit_transform(X_train)
     X_test = vectorizer.transform(X_test)
     clf = MultinomialNB()
     clf.fit(X_train, y_train)
     y_pred = clf.predict(X_test)
     accuracy = (accuracy_score(y_test, y_pred))
     return accuracy

def tokenize(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    tokens = [word for word in nltk.word_tokenize(text) if len(word)>3]
    stems = [stemmer.stem(item) for item in tokens]
    return tokens, stems

def lda_visulaisation(text):
    data = Tweets.find()
    df = pd.DataFrame(data).apply(cleaning_dataset)
    df = df['post_text'].apply(remove_usernames)
    tokens = [word for word in nltk.word_tokenize(text) if len(word)>3]
    stems = [stemmer.stem(item) for item in tokens]
    tokenized_data_strings = [' '.join(tokens) for tokens in df['post_text'] if len(tokens)>3]
    vectorizer_tf = TfidfVectorizer(tokenizer=tokenize, stop_words='english', max_df=0.75, max_features=10000, use_idf=False, norm=None)
    tf_vectors = vectorizer_tf.fit_transform(df.post_text)
    vectorizer = TfidfVectorizer(tokenizer=tokenize, stop_words='english', max_df=0.75, max_features=10000, use_idf=False, norm=None)
    tfidf = vectorizer.fit_transform(tokenized_data_strings)
    dictionary = Dictionary.from_corpus(corpus, id2word={i: w for w, i in vectorizer_tf.vocabulary_.items()})
    corpus = gensim.matutils.Sparse2Corpus(tf_vectors.T)
    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=25, passes=10, alpha='auto')
    vis = gensimvis.prepare(lda_model, corpus, dictionary)
    return pyLDAvis.save_html(vis, '/assets/visualization.html')

@app.route("/api/v1.0/visualisations", methods=["GET", "POST"])
def bertmodel():
    text_list = df["post_text"].tolist()
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    input_ids = []
    attention_masks = []
    for text in text_list:
        encoded_dict = tokenizer.encode_plus(
                        text,                      # Input text
                        add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                        max_length = 64,           # Pad or truncate all sentences
                        pad_to_max_length = True,
                        return_attention_mask = True,   # Construct attn. masks
                        return_tensors = 'pt',     # Return PyTorch tensors
                   )
    
    input_ids.append(encoded_dict['input_ids'])
    attention_masks.append(encoded_dict['attention_mask'])
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    batch_size = 32
    num_batches = (len(input_ids) + batch_size - 1) // batch_size
    predictions = []
    with torch.no_grad():
        for i in range(num_batches):
            start = i * batch_size
            end = min((i + 1) * batch_size, len(input_ids))
            batch_input_ids = input_ids[start:end].to(device)
            batch_attention_masks = attention_masks[start:end].to(device)
            outputs = model(batch_input_ids, attention_mask=batch_attention_masks)
            logits = outputs[0]
            batch_predictions = torch.argmax(logits, dim=1).cpu().numpy()
            predictions.extend(batch_predictions)
    df_pred = pd.DataFrame({'text': text_list, 'predicted_label': predictions})
    grouped = df_pred.groupby('predicted_label').count()
    predicted = grouped.plot(kind='bar', legend=False)
    predicted.set_xlabel('Predicted Label')
    predicted.set_ylabel('Count')
    plt.imshow(predicted)
    plt.savefig('src/assets/bertmodelgraph.png')
    df_pred.to_json('assets/model_output.json', orient='records')

    return render_template('visualisation.component.html', lda_visulaisation=lda_visulaisation, bertmodel=bertmodel )

@app.route("/api/v1.0/discussion", methods=["GET"]) 
def show_discussions():
    page_num, page_size = 1, 10
    if request.args.get('pn'):
        page_num = int(request.args.get('pn'))
    if request.args.get('ps'):
        page_size = int(request.args.get('ps'))
    page_start = (page_size * (page_num - 1))

    data_to_return = []
    for discussion in discussion_forume.find().skip(page_start).limit(page_size):
        discussion['_id'] = str(discussion['_id']) 
        for comment in discussion['comments']:
            comment['_id'] = str(comment['_id']) 
        data_to_return.append(discussion)

    return make_response( jsonify(data_to_return), 200 )

@app.route("/api/v1.0/discussion", methods=["POST"]) 
def add_discussion():
    if "username" in request.form and "title" in request.form and "content" in request.form:
        new_discussion = {
            "username" : request.form["username"],
            "title" : request.form["title"],
            "content" : request.form["content"],    
            "comments" : []
        }

        new_discussion_id = discussion_forume.insert_one(new_discussion)
        new_discussion_link = "http://localhost:5000/api/v1.0/discussion/" + str(new_discussion_id.inserted_id)
        return make_response( jsonify({"url": new_discussion_link} ), 201)
    else:
        return make_response( jsonify({"error":"Missing form data"} ), 404)



@app.route("/api/v1.0/discussion/<string:id>", methods=["GET"]) 
def show_one_discussion(id):
    if len(id) != 24 or not all(c in string.hexdigits for c in id):
        return make_response( jsonify( {"error" : "Invalid discussion ID"} ), 404 )
    discussion = discussion_forume.find_one({'_id':ObjectId(id)}) 
    if discussion is not None:
        discussion['_id'] = str(discussion['_id']) 
        for comment in discussion['comments']:
            comment['_id'] = str(comment['_id'])
        return make_response( jsonify( [discussion] ), 200 )
    else:
        return make_response( jsonify( {"error" : "Invalid discussion ID"} ), 404 )

@app.route("/api/v1.0/discussion/<string:id>", methods=['PUT'])
def edit_discussion(id):
    if "title" in request.form and "content" in request.form:
        result=discussion_forume.update_one({"_id" : ObjectId(id)},{
            "$set" : {"title" : request.form["title"],
                      "content" : request.form["content"]}
        })
        if result .matched_count ==1:
            edited_discussion_link = "http://localhost:5000/api/v1.0/discussion/" + id
            return make_response (jsonify({"url":edited_discussion_link}), 200)
        else:
            return make_response(jsonify({"error":"Invalid Discussion ID"}), 400)
    else: 
        return make_response(jsonify({"error": "Missing Data"}), 400)
    

@app.route("/api/v1.0/discussion/<string:id>", methods=['DELETE'])
def delete_discussion(id):
    result= discussion_forume.delete_one({"_id": ObjectId(id)})
    if result.deleted_count ==1:
        return make_response(jsonify({}), 204)
    else:
        return make_response(jsonify({"error": "Invalid Discussion ID"}), 404)
    

@app.route("/api/v1.0/discussion/<string:id>/comments", methods=['POST'])
def add_comment(id):
    new_comment ={
        "_id" : ObjectId(),
        "username" : request.form["username"],
        "comment" : request.form["comment"],
        "emotions" : request.form['emotions']
    }

    if not "username" or not "comment" or not "emotions": 
        return make_response(jsonify({"error" : "Missing Data"}), 404)
    discussion_forume.update_one({"_id":ObjectId(id)}, {"$push": {"comments": new_comment}})
    new_comment_link = "http://localhost:5000/api/v1.0/discussion/" + id +"/comments/" + str(new_comment['_id'])
    return make_response(jsonify({"url": new_comment_link}), 201)


@app.route("/api/v1.0/discussion/<string:id>/comments", methods=['GET'])
def get_all_comments(id):
    data_to_return =[]
    discussion = discussion_forume.find_one({"_id":ObjectId(id)}, {"comments" : 1, "_id" : 0})
    for comment in discussion["comments"]:
        comment["_id"] = str(comment["_id"])
        data_to_return.append(comment)
    return make_response(jsonify(data_to_return), 200)


@app.route("/api/v1.0/discussion/<string:id>/comments/<string:comment_id>", methods=['GET'])
def get_one_comment(id, comment_id):
    discussion = discussion_forume.find_one({"comments._id" : ObjectId(comment_id)}, {"_id": 0, "comments.$": 1})
    if discussion is None:
        return make_response(jsonify(discussion["comments"][0]), 200)

@app.route("/api/v1.0/discussion/<string:id>/comments/<string:comment_id>", methods=['DELETE'])
def delete_comment(id, comment_id):
    discussion_forume.update_one({"_id": ObjectId(id)}, {"$pull": {"comments": {"_id": ObjectId(comment_id)}}})
    return make_response(jsonify({}), 204)

@app.route('/api/v1.0/count', methods=['GET'])
def count():
   total_count = discussion_forume.count_documents({})
   return str(total_count)


@app.route("/api/v1.0/self-assess", methods=["POST"])
def add_assess():
     if "question1" in request.form and "question2" in request.form and "question3" in request.form and "question4" in request.form and "question5" in request.form and "question6" in request.form and "question6_select" in request.form:
        new_asssessment = {
            "question1" : request.form["question1"],
            "question2" : request.form["question2"],
            "question3" : request.form["question3"],    
            "question4" : request.form["question4"],
            "question5" : request.form["question5"],   
            "question6" : request.form["question6"],   
            "question6_select" : request.form["question6_select"], 
        }

        new_assessment_id = self_assessment.insert_one(new_asssessment)
        new_assessment_link = "http://localhost:5000/api/v1.0/self-assess/" + str(new_assessment_id.inserted_id)
        return make_response( jsonify({"url": new_assessment_link} ), 201)
     else:
        return make_response( jsonify({"error":"Missing form data"} ), 404)
     
@app.route('/api/v1.0/assessmentcount', methods=['GET'])
def asess_count():
   total_count = self_assessment.count_documents({})
   return str(total_count)
     
if __name__ == "__main__":
    app.run(debug=True)

