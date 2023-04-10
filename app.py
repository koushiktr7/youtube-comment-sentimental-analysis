from flask import Flask, render_template, request
import os
import csv
import re
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import pandas as pd; import os
import csv; import numpy as np
import re; import warnings
warnings.filterwarnings('ignore')
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import *
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from textblob import TextBlob
import pandas as pd
import os
import plotly.io as pio
import matplotlib.pyplot as plt
import random; random.seed(5)
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn import metrics
import uuid
from io import BytesIO
import base64




app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    video_id = request.form['video_id']
    
    DEVELOPER_KEY = 'AIzaSyAvPJvvrIZqZVBpETSytkk-EhXpfTWIqU0'
    YOUTUBE_API_SERVICE_NAME = 'youtube'
    YOUTUBE_API_VERSION = 'v3'
    youtube = build('youtube', 'v3', developerKey=DEVELOPER_KEY)

    # Ask the user to enter the YouTube video URL
    url = video_id

    # Extract the video ID from the URL using a regular expression
    match = re.search(r'(?<=v=)[\w-]+|(?<=be/)[\w-]+', url)
    if not match:
        return render_template('result-invalid_url.html', video_id=video_id)

    video_id = match.group(0)
    print(f"The video ID is: {video_id}")
    comments = []
    next_page_token = None
    while True:
        try:
            # Call the API to retrieve the comments
            results = youtube.commentThreads().list(
                part='snippet',
                videoId=video_id,
                maxResults=10000,
                pageToken=next_page_token,
                textFormat='plainText'
            ).execute()

            # Add the comments and their like counts to the list
            for item in results['items']:
                comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
                like_count = item['snippet']['topLevelComment']['snippet']['likeCount']
                comments.append([comment, like_count])

            # Check if there are more pages of comments to retrieve
            next_page_token = results.get('nextPageToken')
            if not next_page_token:
                break

        except HttpError as error:
            print(f'An error occurred: {error}')
            break

    # Write the comments and their like counts to a CSV file
    filename = f"{video_id}_comments.csv"
    with open(filename, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Comment', 'Like Count'])
        for comment in comments:
            writer.writerow(comment)
        print(f"{len(comments)} comments written to {filename}")

         # Read the comments from the CSV file
    df = pd.read_csv(filename)

    # Clean the comments
    for row in range(len(df)):
        line = df.loc[row, "Comment"]
        df.loc[row,"Comment"] = re.sub("[^a-zA-Z]", " ", line)

    sw = stopwords.words('english')
    ps = PorterStemmer()
    lemmatizer = nltk.stem.WordNetLemmatizer()

    def nlpFunction(DF):
    # Convert comments to lowercase and split into tokens
       DF['com_token'] = DF['Comment'].str.lower().str.split()

    # Remove stop words from comments
       DF['com_remv'] = DF['com_token'].apply(lambda x: [y for y in x if y not in sw])

    # Lemmatize the remaining words in comments
       DF["com_lemma"] = DF['com_remv'].apply(lambda x : [lemmatizer.lemmatize(y) for y in x])

    # Stem the lemmatized words in comments
       DF['com_stem'] = DF['com_lemma'].apply(lambda x : [ps.stem(y) for y in x])

    # Join the stemmed words back into a string
       DF["com_tok_str"] = DF["com_stem"].apply(', '.join)

    # Join the remaining words back into a string
       DF["com_full"] = DF["com_remv"].apply(' '.join)

       return DF
    df = nlpFunction(df)

    def drop_cols_after_nlp(Comments):
      Comments = Comments.drop(columns = ['Comment', 'com_token', 'com_remv', 'com_lemma', 'com_stem', 'com_tok_str'], axis = 1)
      return Comments
    df = drop_cols_after_nlp(df)

    df.rename(columns = {'com_full': 'comment'}, inplace=True)
    

# save the updated dataframe to a new csv file
    df.to_csv('with_polarity.csv', index=False)
    def remove_missing_vals(comments): 
      comments['comment'] = comments['comment'].str.strip()
      comments = comments[comments.comment != 'nan'] # remove nan values from data
      comments = comments[comments.comment != '']
    
    remove_missing_vals(df)


    polarity_scores = [] 

    for i, comment in enumerate(df['comment']):
      blob = TextBlob(comment)
      polarity = blob.sentiment.polarity
      polarity_scores.append(polarity)
      polarity_int = round(polarity * 100)
      df.at[i, 'polarity_int'] = polarity_int

# Add the polarity scores to the DataFrame
    df['polarity'] = polarity_scores

# save the updated dataframe to a new csv file
    df.to_csv('with_polarity.csv', index=False)

    df['polarity_int'][df.polarity_int==0]= 0
    df['polarity_int'][df.polarity_int > 0]= 1
    df['polarity_int'][df.polarity_int < 0]= -1
    
    output = df.polarity_int.value_counts().tolist()
    count_list = [[label, count] for label, count in zip(['neutral', 'positive','negative'], output)]

    image_name = str(uuid.uuid1()) + ".png"
    df.polarity_int.value_counts().plot.bar().get_figure().savefig(fname=f"static/{image_name}", format="png")

    df['polarity_int'].isna().sum()
    df = df[df['polarity_int'].notna()]
    df['polarity_int'].isna().sum()

    X = df['comment']
    y = df.polarity_int
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=53, test_size=0.25)

    count_vectorizer = CountVectorizer(stop_words='english', min_df=0.05, max_df=0.9)
    count_train = count_vectorizer.fit_transform(X_train)
    count_test = count_vectorizer.transform(X_test)

    tfidf_vectorizer = TfidfVectorizer(stop_words='english', min_df=0.05, max_df=0.9)
    tfidf_train = tfidf_vectorizer.fit_transform(X_train)
    tfidf_test = tfidf_vectorizer.transform(X_test)
    # Create a MulitnomialNB model
    tfidf_nb = MultinomialNB()
    tfidf_nb.fit(tfidf_train,y_train)
# Run predict on your TF-IDF test data to get your predictions
    tfidf_nb_pred = tfidf_nb.predict(tfidf_test)

# Calculate the accuracy of your predictions
    tfidf_nb_score = metrics.accuracy_score(y_test,tfidf_nb_pred)

# Create a MulitnomialNB model
    count_nb = MultinomialNB()
    count_nb.fit(count_train,y_train)

# Run predict on your count test data to get your predictions
    count_nb_pred = count_nb.predict(count_test)

# Calculate the accuracy of your predictions
    count_nb_score = metrics.accuracy_score(count_nb_pred,y_test)

    print('NaiveBayes Tfidf Score: ', tfidf_nb_score)
    print('NaiveBayes Count Score: ', count_nb_score)

    from sklearn.linear_model import LogisticRegression
    lr_model = LogisticRegression()
    lr_model.fit(tfidf_train,y_train)
    accuracy_lr = lr_model.score(tfidf_test,y_test)
    print("Logistic Regression accuracy is (for Tfidf) :",accuracy_lr)
     
    lr_model = LogisticRegression()
    lr_model.fit(count_train,y_train)
    accuracy_lr = lr_model.score(count_test,y_test)
    print("Logistic Regression accuracy is (for Count) :",accuracy_lr)
    
    # Create a SVM model
    from sklearn import svm
    tfidf_svc = svm.SVC(kernel='linear', C=1)

    tfidf_svc.fit(tfidf_train,y_train)
# Run predict on your tfidf test data to get your predictions
    tfidf_svc_pred = tfidf_svc.predict(tfidf_test)

# Calculate your accuracy using the metrics module
    tfidf_svc_score = metrics.accuracy_score(y_test,tfidf_svc_pred)

    print("LinearSVC Score (for tfidf):   %0.3f" % tfidf_svc_score)
    
    count_svc = svm.SVC(kernel='linear', C=1)

    count_svc.fit(count_train,y_train)
# Run predict on your count test data to get your predictions
    count_svc_pred = count_svc.predict(count_test)

# Calculate your accuracy using the metrics module
    count_svc_score = metrics.accuracy_score(y_test,count_svc_pred)

    print("LinearSVC Score (for Count):   %0.3f" % tfidf_svc_score)
    
    from sklearn.tree import DecisionTreeClassifier
    dt_model = DecisionTreeClassifier()
    dt_model.fit(tfidf_train,y_train)
    accuracy_dt = dt_model.score(tfidf_test,y_test)
    print("Decision Tree accuracy is (for Tfidf):",accuracy_dt)
    
    dt_model = DecisionTreeClassifier()
    dt_model.fit(count_train,y_train)
    accuracy_dt = dt_model.score(count_test,y_test)
    print("Decision Tree accuracy is (for Count):",accuracy_dt)

    from sklearn.ensemble import RandomForestClassifier
    rf_model_initial = RandomForestClassifier(n_estimators = 5, random_state = 1)
    rf_model_initial.fit(tfidf_train,y_train)
    print("Random Forest accuracy for 5 trees is (Tfidf):",rf_model_initial.score(tfidf_test,y_test))

    rf_model_initial = RandomForestClassifier(n_estimators = 5, random_state = 1)
    rf_model_initial.fit(count_train,y_train)
    print("Random Forest accuracy for 5 trees is (Count):",rf_model_initial.score(count_test,y_test))

    tfidf_pred = tfidf_vectorizer.transform(df['comment'])
    tfidf_svc_pred = tfidf_svc.predict(tfidf_pred)

    neutral = (tfidf_svc_pred == 0.0).sum()
    positive = (tfidf_svc_pred == 1.0).sum()
    negative = (tfidf_svc_pred < 0).sum()

   
    result_dict = {
    'neutral': int((tfidf_svc_pred == 0.0).sum()),
    'positive': int((tfidf_svc_pred == 1.0).sum()),
    'negative': int((tfidf_svc_pred < 0).sum())
    }

    result_list = [[k, v] for k, v in result_dict.items()]

    print(result_list)


# Your sentiment analysis code here
    total_comments = len(df)
    positive_percent = (positive / total_comments) * 100
    negative_percent = (negative / total_comments) * 100
    neutral_percent = (neutral / total_comments) * 100

# Create a bar chart
    labels = ['Positive', 'Negative', 'Neutral']
    values = [positive_percent, negative_percent, neutral_percent]
    colors = ['#00FF00', '#FF0000', '#0000FF']
   
# Generate the plot as an image
    fig,ax = plt.subplots()
    ax.bar(labels, values, color=colors, width=0.4)
    for i, v in enumerate(values):
       ax.text(i, v+2, f"{v:.2f}%", ha='center', fontsize=14)
    if positive_percent > negative_percent:
       ax.text(0.5, -14, "The video is good!", ha='center', fontsize=16, color='green')
    else:
        ax.text(0.5, -14, "The video is bad.", ha='center', fontsize=16, color='red')

    ax.set_title('Sentiment Analysis', fontsize=18)
    ax.tick_params(axis='both', labelsize=14)
    plt.tight_layout()

# Save the plot as a PNG image
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)

# Encode the image in base64 format for embedding in HTML
    img_str = base64.b64encode(buffer.read()).decode()
    
  
    return render_template('result.html', 
                           video_id=video_id, 
                           polarity_out=count_list,
                           image = image_name,video_result=result_list, 
                           img_data=img_str)
  
    

if __name__ == '__main__':
    app.run(debug=True, port=8000)     