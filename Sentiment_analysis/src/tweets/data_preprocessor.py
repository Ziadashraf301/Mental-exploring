from Sentiment_analysis.src.logger.train_logger import get_logger
import re
import nltk

nltk.download('punkt_tab')
nltk.download('wordnet')

from nltk.stem import WordNetLemmatizer
import warnings
warnings.filterwarnings('ignore')

LOGGER = get_logger()

# Defining dictionary containing all emojis with their meanings.
emojis = {':)': 'smile', ':-)': 'smile', ';d': 'wink', ':-E': 'vampire', ':(': 'sad', 
          ':-(': 'sad', ':-<': 'sad', ':P': 'raspberry', ':O': 'surprised',
          ':-@': 'shocked', ':@': 'shocked',':-$': 'confused', ':\\': 'annoyed', 
          ':#': 'mute', ':X': 'mute', ':^)': 'smile', ':-&': 'confused', '$_$': 'greedy',
          '@@': 'eyeroll', ':-!': 'confused', ':-D': 'smile', ':-0': 'yell', 'O.o': 'confused',
          '<(-_-)>': 'robot', 'd[-_-]b': 'dj', ":'-)": 'sadsmile', ';)': 'wink', 
          ';-)': 'wink', 'O:-)': 'angel','O*-)': 'angel','(:-D': 'gossip', '=^.^=': 'cat'}

def preprocess_text(text):
    # Create Lemmatizer
    wordLemm = WordNetLemmatizer()
    
    # Defining regex patterns.
    urlPattern        = r"((http://)[^ ]*|(https://)[^ ]*|( www\.)[^ ]*)"
    userPattern       = '@[^\s]+'
    alphaPattern      = "[^a-zA-Z0-9]"
    sequencePattern   = r"(.)\1\1+"
    seqReplacePattern = r"\1\1"

    tweet = text.lower()
        
    # Replace all URls with 'URL'
    tweet = re.sub(urlPattern,'',tweet)

    # Replace all emojis.
    for emoji in emojis.keys():
        tweet = tweet.replace(emoji,emojis[emoji])        

    # Replace @USERNAME to 'USER'.
    tweet = re.sub(userPattern,'', tweet)

    # Replace all non alphabets.
    tweet = re.sub(alphaPattern, " ", tweet)

    # Replace 3 or more consecutive letters by 2 letter.
    tweet = re.sub(sequencePattern, seqReplacePattern, tweet)

    tweetword = ''
    for word in tweet.split():
        if len(word)>1:
        # Lemmatizing the word.
            word = wordLemm.lemmatize(word)
            tweetword += (word+' ')
        
    return tweetword