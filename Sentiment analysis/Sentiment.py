import joblib
import numpy as np
import pandas as pd
from preprocessing import preprocess

def load_models():
    '''
    Replace '..path/' by the path of the saved models.
    '''
    LRmodel = joblib.load('Sentiment-LR.pkl')
    vectoriser = joblib.load('vectoriser.pkl')
    return vectoriser,LRmodel

def predict(vectoriser ,model, text):
    # Predict the sentiment
    textdata = vectoriser.transform(preprocess(text))
    sentiment = model.predict(textdata)
    prop = model.predict_proba(textdata)
    probabilty_negative = prop[:,0]
    probabilty_positive = prop[:,1]

    # Make a list of text with sentiment.
    data = []
    for text, pred,negative,positive in zip(text, sentiment,probabilty_negative,probabilty_positive):
        data.append((text,pred,negative,positive))
        
    # Convert the list into a Pandas DataFrame.
    df = pd.DataFrame(data, columns = ['text','sentiment','Negative','Positive'])
    df = df.replace([0,1], ["Negative","Positive"])
    return df

vectoriser,LRmodel = load_models()

text = ['I am so sad','The product is bad',"I am happy", "He is honest man.", "I have a terrible day"," the product is horrible", "I love her" , "I can't understand"
        ,"I am disapointed" ,"The product ugly","This man is good but i didn't love him","I am so sad","I am so sad and not happy","This food is healthy but has a bad taste"
        ,"I hate twitter","May the Force be with you.","I don't feel so good"]

df = predict(vectoriser,LRmodel, text)
print(df)