#importing dependencies

import pandas as pd
import matplotlib.pyplot as plt
import nltk
import re #regular expression library
import string

#natural language toolkit (nltk) library for sentiment analysis
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords # stop words are commonly used words like a, an, the etc.

#for visual representation of text data, use wordcloud
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

data = pd.read_csv("filename.csv")
print('...printing dataset head...')
print(data.head())

##########################################
print('...printing dataset columns...')
print(data.columns)

###########################################
#using only the below 3 columns
data = data[["username", "tweet", "language"]]

#to check if any of the columns have null values
print('...checking if any column has null value...')
data.isnull().sum()
print('...language count...')
data["language"].value_counts()

################################################

#removing all the links, punctuation, symbols and other language errors from the tweets:

nltk.download('stopwords') #stopwords are the 40 most common words like 'a, an the etc.' that do not provide significance to the analysis

#stemming is reducing a word to its base word or stem in such a way that the words of similar kind lie under a common stem. 
#For example – The words care, cared and caring lie under the same stem ‘care’. 
stemmer = nltk.SnowballStemmer("english")
stopword=set(stopwords.words('english'))

def clean(text):
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = [word for word in text.split(' ') if word not in stopword]
    text=" ".join(text)
    text = [stemmer.stem(word) for word in text.split(' ')]
    text=" ".join(text)
    return text
data["tweet"] = data["tweet"].apply(clean)

###########################################

#adding columns positive, negative and neutral in the dataset
print('...adding columns positive, negative and neutral in the dataset...')
nltk.download('vader_lexicon')
sentiments = SentimentIntensityAnalyzer()
data["Positive"] = [sentiments.polarity_scores(i)["pos"] for i in data["tweet"]]
data["Negative"] = [sentiments.polarity_scores(i)["neg"] for i in data["tweet"]]
data["Neutral"] = [sentiments.polarity_scores(i)["neu"] for i in data["tweet"]]
data = data[["tweet", "Positive", "Negative", "Neutral"]]
print(data.head())

######################################

#add the code here from readme.md for visualising positive, negative or most commonly used words in the tweet as per the choice
