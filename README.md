
** Tweet_Analyzer**
**Using python packages for sentiment analysis of the Russia Vs Ukraine War tweets:**
1. nltk 
2. pandas
3. matplotlib
4. wordcloud
5. re
6. string

**Dataset has been downloaded from Kaggle:**
https://www.kaggle.com/datasets/towhidultonmoy/russia-vs-ukraine-tweets-datasetdaily-updated?resource=download

#############################

**add the below codes into the python file as per the choice:**

1. For viewing most frequently used words in the tweets 
-------------------------------------------------------
text = " ".join(i for i in data.tweet)
stopwords = set(STOPWORDS)
wordcloud = WordCloud(stopwords=stopwords, background_color="white").generate(text)
plt.figure( figsize=(15,10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

2. For viewing most frequently used words by positive commenters
----------------------------------------------------------------
positive =' '.join([i for i in data['tweet'][data['Positive'] > data["Negative"]]])
stopwords = set(STOPWORDS)
wordcloud = WordCloud(stopwords=stopwords, background_color="white").generate(positive)
plt.figure( figsize=(15,10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

3. For viewing most frequently used words by negative commneters
--------------------------------------------------------
negative =' '.join([i for i in data['tweet'][data['Negative'] > data["Positive"]]])
stopwords = set(STOPWORDS)
wordcloud = WordCloud(stopwords=stopwords, background_color="white").generate(negative)
plt.figure( figsize=(15,10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
