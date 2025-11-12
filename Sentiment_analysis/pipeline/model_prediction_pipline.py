import joblib
from helpers.text_preprocessor import preprocess_text

def load_models():
    LRmodel = joblib.load('models/Sentiment-LR.pkl')
    vectoriser = joblib.load('models/vectoriser.pkl')
    return vectoriser,LRmodel

def predict(vectoriser ,model, text):
    # Predict the sentiment
    textdata = vectoriser.transform([preprocess_text(text)])
    sentiment = model.predict(textdata)[0]
    prop = model.predict_proba(textdata)
    probabilty_negative = prop[:,0][0]
    probabilty_positive = prop[:,1][0]

    return sentiment, probabilty_negative, probabilty_positive


def main():
    vectoriser,LRmodel = load_models()

    tweets = ['I am so sad',
            'The product is bad',
            "I am happy", 
            "He is honest man.", 
            "I have a terrible day",
            " the product is horrible", 
            "I love her", 
            "I can't understand",
            "I am disapointed" ,
            "The product ugly",
            "This man is good but i didn't love him",
            "I am so sad","I am so sad and not happy",
            "This food is healthy but has a bad taste",
            "I hate twitter",
            "May the Force be with you.",
            "I don't feel so good"]

    for tweet in tweets:
        sentiment, probabilty_negative, probabilty_positive = predict(vectoriser,LRmodel, tweet)
        print(tweet, sentiment, probabilty_negative, probabilty_positive)


if __name__== '__main__':
    main()