Hi :wave:,

In this repository, I used Natural Language Processing and create a model that predicts whether a tweet is related with disasters or not. This dataset is belongs to Kaggle competition.
[Natural Language Processing with Disaster Tweets](https://www.kaggle.com/c/nlp-getting-started/overview)

:loudspeaker: In exploratory data analysis I took tweets and target column and applied word_tokenize, removed stopwords and applied lemmatize.

![Function](https://github.com/M-Rasit/NLP-Disaster-Tweets/blob/master/images/function.png?raw=true)

:loudspeaker: Target column is binary. Disaster and Non-disaster. This is a balanced dataset.

![Target](https://github.com/M-Rasit/NLP-Disaster-Tweets/blob/master/images/target.png?raw=true)

:loudspeaker: "http" and "https" are common words for both target values. So that I removed them. I created wordcloud for both values.

![Wordcloud](https://github.com/M-Rasit/NLP-Disaster-Tweets/blob/master/images/wordcloud.png?raw=true)

:loudspeaker: I transfomed X by both CountVectorizer and TfidfVectorizer and apllied various Machine Learning Models. Multinomial Naive Bayes transformed with CountVectorizer performed best f1 score.

![F1 Score](https://github.com/M-Rasit/NLP-Disaster-Tweets/blob/master/images/f1_score.png?raw=true)

:loudspeaker: Then I created web app by using Streamlit. 

![Streamlit](https://github.com/M-Rasit/NLP-Disaster-Tweets/blob/master/images/tweets_streamlit.png?raw=true)

Thank you :tulip:
