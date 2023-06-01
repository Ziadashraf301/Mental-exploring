import string
import re
import preprocessor as p
from  nltk.tokenize import word_tokenize 
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

#HappyEmoticons
emoticons_happy = set([
    ':-)', ':)', ';)', ':o)', ':]', ':3', ':c)', ':>', '=]', '8)', '=)', ':}',
   ':^)', ':-D', ':D', '8-D', '8D', 'x-D', 'xD', 'X-D', 'XD', '=-D', '=D','=-3', '=3', ':-))',
    ":'-)", ":')", ':*', ':^*', '>:P', ':-P', ':P', 'X-P','x-p', 'xp', 'XP', ':-p', ':p', '=p',
    ':-b', ':b', '>:)', '>;)', '>:-)','<3'
    ])

# Sad Emoticons
emoticons_sad = set([
    ':L', ':-/', '>:/', ':S', '>:[', ':@', ':-(', ':[', ':-||', '=L', ':<',
    ':-[', ':-<', '=\\', '=/', '>:(', ':(', '>.<', ":'-(", ":'(", ':\\', ':-c',
    ':c', ':{', '>:\\', ';('
    ])

#Emoji patterns
emoji_pattern = re.compile("["
         u"\U0001F600-\U0001F64F"  # emoticons
         u"\U0001F300-\U0001F5FF"  # symbols & pictographs
         u"\U0001F680-\U0001F6FF"  # transport & map symbols
         u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
         u"\U00002702-\U000027B0"
         u"\U000024C2-\U0001F251"
         "]+", flags=re.UNICODE)


# words difficult to detect by the preprocessing modules
difficult_to_detect = ["'re","'s","'m","'ve","n't","...","``","'","im",
    "ca","itv","-","a.","dont","us","could","can","'d","__",'aaron', 'ab', 
    'zurab', 'zwart', 'zyl',"ll","u",'__', '___', '____','a', 'about', 'above', 'after', 'again', 'ain', 'all', 'am', 'an',
             'and','any','are', 'as', 'at', 'be', 'because', 'been', 'before',
             'being', 'below', 'between','both', 'by', 'can', 'd', 'did', 'do',
             'does', 'doing', 'down', 'during', 'each','few', 'for', 'from', 
             'further', 'had', 'has', 'have', 'having', 'he', 'her', 'here',
             'hers', 'herself', 'him', 'himself', 'his', 'how', 'i', 'if', 'in',
             'into','is', 'it', 'its', 'itself', 'just', 'll', 'm', 'ma',
             'me', 'more', 'most','my', 'myself', 'now', 'o', 'of', 'on', 'once',
             'only', 'or', 'other', 'our', 'ours','ourselves', 'out', 'own', 're',
             's', 'same', 'she', "shes", 'should', "shouldve",'so', 'some', 'such',
             't', 'than', 'that', "thatll", 'the', 'their', 'theirs', 'them',
             'themselves', 'then', 'there', 'these', 'they', 'this', 'those', 
             'through', 'to', 'too','under', 'until', 'up', 've', 'very', 'was',
             'we', 'were', 'what', 'when', 'where','which','while', 'who', 'whom',
             'why', 'will', 'with', 'won', 'y', 'you', "youd","youll", "youre",
             "youve", 'your', 'yours', 'yourself', 'yourselves']

difficult_to_detect = set(difficult_to_detect)


#combine sad and happy emoticons
emoticons = emoticons_happy.union(emoticons_sad)

lemmatizer = WordNetLemmatizer()

def clean_tweets(tweet):

    tweet = tweet.lower() #lower the words to be in the same format for the modules

    tweet = re.sub (r':', '', tweet) #after tweepy preprocessing the colon symbol left remain after

    tweet = re.sub (r',ÄI', '', tweet) #removing mentions
    
    tweet = re.sub (r'[^\x00-\x7F]+','', tweet) #replace consecutive non-ASCII characters with a space
    
    tweet = emoji_pattern.sub (r'', tweet)  #remove emojis from tweet
    
    tweet = re.sub('[0-9]+', '', tweet) #remove numbers

    tweet = re.sub(f'[{string.punctuation}]','',tweet) #remove punctuation 
    
    stop_words = set(stopwords.words('english')) #get the stop words

    word_tokens = word_tokenize(tweet) #extract the tokens from string of characters
    

    filtered_tweet = [] 

    #looping through conditions to filter the words
    for w in word_tokens:
        #check tokens against stop words, emoticons and words difficult to detect 
        if w not in stop_words and w not in emoticons and w not in difficult_to_detect:
            if len(w)>1: #remove the word if it less than 2 character
                w = lemmatizer.lemmatize(w) # Applay lemmatization on the word 
                filtered_tweet.append(w) #Append the pure word to the list after cleaning

    return ' '.join(filtered_tweet) #Reconstruct the tweet after cleaning
