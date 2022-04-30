#!/usr/bin/env python
# coding: utf-8

# In[77]:
# Import general libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import re
from pandas.core.frame import DataFrame


# Import sentiment analysis libraries
from textblob.classifiers import NaiveBayesClassifier
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import tweepy as tw
from tweepy import OAuthHandler
from nltk.tag import pos_tag
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
import string
import nltk
from nltk.tokenize import word_tokenize
from textblob import TextBlob
from collections import Counter

# Import streamlit 
import streamlit as st

# In[2]:
data = pd.read_csv('bfml.csv')

def cleanTxt(text):

    text = re.sub(r'@[A-Za-z0-9]+','',text)
    text = re.sub(r':', '', text)
    text = re.sub(r'#', '',text)
    text = re.sub(r'RT[\s]+', '',text)
    text = re.sub(r'https?:\/\/\S+', '',text)
    return text

data['tweet_text'] = data['tweet_text'].apply(cleanTxt)

print(data)

# In[3]:
data.isnull().sum()

# In[4]:
data.label.value_counts()

# In[6]:
data.label = data.label.replace({'Positive': 'positive',
                                'Postive': 'positive',
                                'postive': 'positive',
                                'positve': 'positive',
                                'poitive': 'positive',
                                'poistive': 'positive',
                                
                                'Negative': 'negative',
                                 
                                'Neutral': 'neutral'})

# In[7]:
data.label.value_counts()

# In[8]:
data = data[(data['label'] == 'positive') | (data['label'] == 'negative') | (data['label'] == 'neutral')]
data = data[['tweet_text', 'label']]

# In[9]:
data['tweet_text'] = data['tweet_text'].astype(str)
data['label'] = data['label'].astype(str)

# In[10]:
data['data_col'] = data[['tweet_text', 'label']].apply(tuple, axis=1)

# In[12]:
traindf = data[:1100]

# In[13]:
testdf = data[1100:]

# In[14]:
train = traindf['data_col']
train = train.values.tolist()
testdf = testdf['data_col']
testdf = testdf.values.tolist()

print('Training model....')
# In[15]:
cl = NaiveBayesClassifier(train)
print('Done training....')


# App
def main():
    

    # Add Navigation bar
    st.sidebar.title("Black Friday Tweets: An Exploratory Sentiment Analysis")
    st.sidebar.markdown("This application is a Dashboard for sentiment analysis on Black Friday tweets")
    activities = ["Project Description", "Sentiment Analysis"]
    choices = st.sidebar.selectbox("Select Activities",activities)
    background = open("files/background.txt","r")
    description = open("files/description.txt","r")
    goal = open("files/goal.txt","r")



    # Working from Sidebar
    if choices == "Project Description":
        st.title("Sentiment Analysis on Black Friday Tweets")
        st.markdown("Team 53001")
        # st.info("https://github.com/")
        st.header("Background")
        st.write(background.read())
        st.header("Project Description")    
        st.write(description.read())
        st.header("Project Goal")
        st.write(goal.read())

 #################################################################################################################
    # Data Exploration
    elif choices == "Sentiment Analysis":
        
        st.title("Sentiment Analysis")
        st.header("Fetch new black friday tweets for analysis")

        api_key = ['Yjc72iLaViJKBgUkgQTlbxinB']
        api_secret_key = ['e94PdXwuFOuwU8Vh0nveP0MiWlu2NenKH8sdGMiLIoIkn1v2IJ']
        access_token = ['898536142852718592-JrOTEmiLUWaJTzKVw1kVZbk27YqBtnK']
        access_token_secret =['uWV4eQf0rDqzA8idPs1nXRo1l1NiAnegUZcaPcCjAZoDZ']

        # In[23]:
        auth = tw.OAuthHandler('Yjc72iLaViJKBgUkgQTlbxinB','e94PdXwuFOuwU8Vh0nveP0MiWlu2NenKH8sdGMiLIoIkn1v2IJ')
        auth.set_access_token('898536142852718592-JrOTEmiLUWaJTzKVw1kVZbk27YqBtnK','uWV4eQf0rDqzA8idPs1nXRo1l1NiAnegUZcaPcCjAZoDZ')
        api = tw.API(auth, wait_on_rate_limit=True)


        # In[25]:
        tweets = []
        date = st.text_input("Enter date")
        st.markdown("Date must be in the format (YYYY-MM-DD)")
        keyword = st.text_input("Enter keyword(s) to query", value="black tweets")
        st.markdown("Double or Triple words should be separated with spaces")
        count_number = st.number_input("Enter number of tweets you want to query", min_value=10, max_value=1500, step=5)
        st.markdown("Numbers should not include points")

        def tweets_query(api,query,count,max_requests):
            tweets = tw.Cursor(api.search_tweets, q=query,lang="en",since_id=date).items(count)
            #tweets
            tweets_list  = [[tweet.text, tweet.id, tweet.source, tweet.coordinates, tweet.retweet_count, tweet.favorite_count,
                        tweet.user._json['name'],tweet.user._json['screen_name'], tweet.user._json['location'], tweet.user._json['friends_count'],
                        tweet.user._json['verified'], tweet.user._json['description'], tweet.user._json['followers_count']] for tweet in tweets]
            tweets_df= pd.DataFrame(tweets_list, columns = ['tweet_text','tweet_id', 'tweet_source','coordinates','retweet_count','likes_count','Username', 'screen_name','location', 'friends_count','verification_status','description','followers_count'])
            return tweets_df

        def cleanTxt(text):

            text = re.sub(r'@[A-Za-z0-9]+','',text)
            text = re.sub(r':', '', text)
            text = re.sub(r'#', '',text)
            text = re.sub(r'RT[\s]+', '',text)
            text = re.sub(r'https?:\/\/\S+', '',text)
            return text

    

        # In[26]:
        query = keyword
        count = count_number
        max_requests = 3
        print('Fetching most recent 1500 tweets ...')
        tweets= tweets_query(api,query,count,max_requests)
        print('Done fetching most recent 1500 tweets ...')

        # In[32]:
        tweetdf = (tweets
        .sort_values(['followers_count'], ascending=False)
        [['tweet_text', 'retweet_count', 'followers_count']])

        # In[33]:
        tweetdf['tweet_text'] = tweetdf['tweet_text'].astype(str)

        # In[35]:
        inference = tweetdf['tweet_text']
        inference = inference.values.tolist()

        # In[42]:
        sentiments = []
        for tweet in inference:
            sentiment = cl.classify(tweet)
            sentiments.append(sentiment) 

        # In[51]:
        tweetdf['sentiments'] = np.asarray(sentiments)

        # In[62]:
        tweetdfdict  = tweetdf['sentiments'].value_counts().to_dict()

        # In[68]:
        fig, axis = plt.subplots(1,1, figsize = (15,10))
        plt.bar(list(tweetdfdict.keys()), list(tweetdfdict.values()))
        axis.set_xlabel('Sentiments', fontsize=12)
        axis.set_ylabel('Total Number', fontsize=12)
        axis.set_title('Sentiment Analysis on Black Friday tweets', fontsize=14)
        st.pyplot(fig)

        # In[73]:
        sentiment_type = ["positive", "negative"]
        choices = st.sidebar.selectbox("Select sentiment type",sentiment_type)
        if choices ==  "positive":
            pos_review = tweetdf[tweetdf['sentiments'] == 'positive']

            # In[74]:
            text = " ".join(tweet for tweet in pos_review.tweet_text.astype(str))

            # In[75]:
            # Generating the word cloud image
            stopwords=set(STOPWORDS)
            wordcloud_pos = WordCloud(stopwords=stopwords, background_color='white').generate(text)

            # plt.figure(figsize=[20,10])
            fig, ax = plt.subplots()
            st.write(plt.imshow(wordcloud_pos, interpolation='bilinear'))
            st.write(plt.axis("off"))
            st.pyplot(fig)

            # Tabular word frequency
            # In[93]:

            # stop = set(stopwords.words('english'))
            newStopWords = ['rt','RT', '@', 'black', 'friday', 'https', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', 
                            '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30']
            stopwords.update(newStopWords)

            # In[94]:
            pos_review["tweet_text_stop"] = pos_review["tweet_text"].str.replace("[^\w\s]", "").str.lower()


            # In[95]:
            pos_review["tweet_text_stop"] = pos_review["tweet_text_stop"].apply(lambda x: ' '.join([item for item in x.split() if item not in stopwords]))
            # a = pos_review["tweet_text_stop"].str.split(expand=True).stack().value_counts()

            # In[96]:
            # Enabling Cache
            @st.cache(persist=True, show_spinner=True, suppress_st_warning=True)
            def pos_count_loder():
                c = Counter(" ".join(pos_review["tweet_text_stop"].str.lower()).split()).most_common(100)
                df = pd.DataFrame(c, columns = ['word', 'count'])
                st.write(df)
            pos_count_loder()

            print('Positive done')

        if choices ==  "negative":
            pos_review = tweetdf[tweetdf['sentiments'] == 'negative']

            # In[74]:
            text = " ".join(tweet for tweet in pos_review.tweet_text.astype(str))

            # In[75]:
            # Generating the word cloud image
            stopwords=set(STOPWORDS)
            wordcloud_pos = WordCloud(stopwords=stopwords, background_color='white').generate(text)

            # plt.figure(figsize=[20,10])
            fig, ax = plt.subplots()
            st.write(plt.imshow(wordcloud_pos, interpolation='bilinear'))
            st.write(plt.axis("off"))
            st.pyplot(fig)

            # Tabular word frequency
            # In[93]:

            # stop = set(stopwords.words('english'))
            newStopWords = ['rt','RT', '@', 'https', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', 
                            '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30']
            stopwords.update(newStopWords)

            # In[94]:
            pos_review["tweet_text_stop"] = pos_review["tweet_text"].str.replace("[^\w\s]", "").str.lower()


            # In[95]:
            pos_review["tweet_text_stop"] = pos_review["tweet_text_stop"].apply(lambda x: ' '.join([item for item in x.split() if item not in stopwords]))
            # a = pos_review["tweet_text_stop"].str.split(expand=True).stack().value_counts()

            # In[96]:
            @st.cache(persist=True, show_spinner=True, suppress_st_warning=True)
            def neg_count_loder():
                c = Counter(" ".join(pos_review["tweet_text_stop"].str.lower()).split()).most_common(100)
                df = pd.DataFrame(c, columns = ['word', 'count'])
                st.write(df)
            data = neg_count_loder()

            print('Negative done')
        
#############################################################################################################

if __name__ == "__main__":
    main()
