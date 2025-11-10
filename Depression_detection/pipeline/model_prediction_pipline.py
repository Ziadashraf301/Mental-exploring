import numpy as np
import joblib
from helpers.text_preprocessor import clean_tweets
import concurrent.futures
import warnings
warnings.filterwarnings('ignore')

#Load the model from the file
mnb = joblib.load('models/mnb_v2.pkl')

#Load the model from the file
sgd = joblib.load('models/sgd_v2.pkl')

def prediction(new_data, model1, model2):

    """
    This function used for transform the new data and prediction.

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

    # Use ThreadPoolExecutor to make the two predictions in parallel
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future1 = executor.submit(model1.predict, [new_data])
        future2 = executor.submit(model2.predict, [new_data])

    # Retrieve the predicted labels from the Future objects
    predictions1 = future1.result()
    predictions2 = future2.result()

    # Calculate the probabilities for each label using predict_proba method
    label = ['Not depressed','Depressed']
    probabilities1 = np.array(model1.predict_proba([new_data]))
    probabilities2 = np.array(model2.predict_proba([new_data]))

    # Calculate the average of the two probabilities
    avg_pop = (probabilities1 + probabilities2)/2

    # Determine the final prediction label based on the average probability
    if avg_pop[0][1] >= 0.7:
        predic = label[1]

    else:
        predic = label[0]

    print("Probability Depressed {} and Probability not Depressed {}, the user is {}".format(avg_pop[0][1],avg_pop[0][0],predic))

    # Return the final prediction probability and the class
    return avg_pop[0][1],avg_pop[0][0],predic