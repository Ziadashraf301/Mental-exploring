import numpy as np
import joblib
from helpers.text_preprocessor import clean_tweets
import warnings
warnings.filterwarnings('ignore')

#Load the models from the file
mnb = joblib.load('models/mnb_v2.pkl')
sgd = joblib.load('models/sgd_v2.pkl')

def predict_depression(new_data, model1, model2):

    """
    This function used for preprocess the new data and prediction.

    Parameters
    ----------
    new_data : The text to predict its label
    model1 : The first pretrained model
    model2: The second pretrained model

    Returns
    -------
    Return the final prediction probabilities and the class

    """
    # Transform the input data using the clean_tweets function
    new_data = clean_tweets(new_data)

    # Retrieve the predicted labels from the Future objects
    probabilities1 = np.array(model1.predict_proba([new_data]))
    probabilities2 = np.array(model2.predict_proba([new_data]))

    # Calculate the average of the two probabilities
    avg_pop = (probabilities1 + probabilities2) / 2

    # Determine the final prediction label based on the average probability
    label = ['Not depressed','Depressed']
    if avg_pop[0][1] >= 0.7:
        prediction = label[1]
    else:
        prediction = label[0]

    # Return the final prediction probability and the class
    return avg_pop[0][1], avg_pop[0][0], prediction


def main():

    texts = ['I am so sad',
            'The product is bad',
            "I am happy", 
            "He is honest man.", 
            "I have a terrible day",
            " the product is horrible", 
            "I love her" , 
            "I can't understand"
            ,"I am disapointed" ,
            "The product ugly",
            "This man is good but i didn't love him",
            "I am so sad","I am so sad and not happy",
            "This food is healthy but has a bad taste",
            "I hate twitter",
            "May the Force be with you.",
            "I don't feel so good"]

    for text in texts:
        probabilty_positive, probabilty_negative, prediction = predict_depression(text, mnb, sgd)
        print(text, prediction, probabilty_negative, probabilty_positive)


if __name__ == "__main__":
    main()