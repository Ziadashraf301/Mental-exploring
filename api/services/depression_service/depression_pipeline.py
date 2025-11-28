"""
Production Inference Pipeline for Depression Detection using BERT
Loads model from Hugging Face Hub
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os
import json
from datetime import datetime
import logging
from config import settings
import re
import nltk
import string
from  nltk.tokenize import word_tokenize

nltk.download('punkt_tab', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('stopwords', quiet=True)

from nltk.stem import WordNetLemmatizer


class DepressionDetectionPipeline:
    def __init__(self):
        # Load configuration
        self.CONFIG = settings

        # Setup logger
        logging.basicConfig(
            level=getattr(logging, self.CONFIG.DEPRESSION_LOG_LEVEL),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.CONFIG.DEPRESSION_LOG_FILE, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.LOGGER = logging.getLogger(__name__)

        # Initialize BERT model and tokenizer
        self.MODEL = None
        self.TOKENIZER = None
        self.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def initialize_pipeline(self):
        """
        Initialize the inference pipeline
        Load BERT model and tokenizer from Hugging Face
        """
        self.LOGGER.info("INITIALIZING DEPRESSION DETECTION INFERENCE PIPELINE")
        self.LOGGER.info(f"Using device: {self.DEVICE}")

        # Load tokenizer
        self.LOGGER.info(f"Loading tokenizer from: {self.CONFIG.DEPRESSION_MODEL_NAME}")
        try:
            self.TOKENIZER = AutoTokenizer.from_pretrained(
                self.CONFIG.DEPRESSION_MODEL_NAME,
                token=self.CONFIG.DEPRESSION_MODEL_HUGGINGFACE_TOKEN if hasattr(self.CONFIG, 'DEPRESSION_MODEL_HUGGINGFACE_TOKEN') else None
            )
            self.LOGGER.info(f"✓ Tokenizer loaded successfully")
        except Exception as e:
            self.LOGGER.error(f"✗ Failed to load tokenizer: {str(e)}")
            raise

        # Load model
        self.LOGGER.info(f"Loading model from: {self.CONFIG.DEPRESSION_MODEL_NAME}")
        try:
            self.MODEL = AutoModelForSequenceClassification.from_pretrained(
                self.CONFIG.DEPRESSION_MODEL_NAME,
                token=self.CONFIG.DEPRESSION_MODEL_HUGGINGFACE_TOKEN if hasattr(self.CONFIG, 'DEPRESSION_MODEL_HUGGINGFACE_TOKEN') else None
            )
            self.MODEL.to(self.DEVICE)
            self.MODEL.eval()
            self.LOGGER.info(f"✓ Model loaded successfully and set to eval mode")
        except Exception as e:
            self.LOGGER.error(f"✗ Failed to load model: {str(e)}")
            raise

        self.LOGGER.info("PIPELINE INITIALIZATION COMPLETE")

    def preprocess_text(self, text):
        """
        Basic text preprocessing (optional for BERT - it can handle raw text well)
        Keep minimal preprocessing to preserve context
        """
        try:
            # Create Lemmatizer
            lemmatizer = WordNetLemmatizer()
            
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
                    u"\U0001F600-\U0001F64F"  
                    u"\U0001F300-\U0001F5FF"
                    u"\U0001F680-\U0001F6FF"
                    u"\U0001F1E0-\U0001F1FF"
                    u"\U00002702-\U000027B0"
                    u"\U000024C2-\U0001F251"
                    "]+", flags=re.UNICODE)


            # words difficult to detect by the preprocessing modules
            difficult_to_detect = ["'re","'s","'m","'ve","n't","...","``","'","im",
                "ca","itv","-","a.","dont","us","could","can","'d","__",'aaron', 'ab', 
                'zurab', 'zwart', 'zyl',"ll","u",'__', '___', '____','a', 'about', 'above', 
                'after', 'again', 'ain', 'all', 'am', 'an',
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

            text = text.lower() #lower the words to be in the same format for the modules

            text = re.sub (r':', '', text) #after tweepy preprocessing the colon symbol left remain after

            text = re.sub (r',ÄI', '', text) #removing mentions
            
            text = re.sub (r'[^\x00-\x7F]+','', text) #replace consecutive non-ASCII characters with a space
            
            text = emoji_pattern.sub (r'', text)  #remove emojis from tweet
            
            text = re.sub('[0-9]+', '', text) #remove numbers

            text = re.sub(f'[{string.punctuation}]','',text) #remove punctuation 

            stop_words = set(stopwords.words('english')) #get the stop words
            
            word_tokens = word_tokenize(text) #extract the tokens from string of characters
            
            preprocessed_text = [] 

            #looping through conditions to filter the words
            for w in word_tokens:
                #check tokens against stop words, emoticons and words difficult to detect 
                if w not in stop_words and w not in emoticons and w not in difficult_to_detect:
                    if len(w)>1: #remove the word if it less than 2 character
                        w = lemmatizer.lemmatize(w) # Applay lemmatization on the word 
                        preprocessed_text.append(w) #Append the pure word to the list after cleaning

            Text = ' '.join(preprocessed_text) #Reconstruct the tweet after cleaning
            
            self.LOGGER.info(f"✓ Text preprocessing complete")
        
        except Exception as e:
            self.LOGGER.error(f"✗ Text preprocessing failed: {str(e)}")
            raise
            
        return Text.strip()

    def predict_depression(self, text, save_result=False):
        """
        Predict depression from text using BERT model
        
        Parameters:
        -----------
        text : str
            Text to analyze
        save_result : bool
            Whether to save results to JSON
            
        Returns:
        --------
        dict
            Prediction results including probabilities
        """
        try:
            if not text:
                self.LOGGER.warning("No text provided for depression detection")
                result = {
                    "success": True,
                    "timestamp": datetime.now().isoformat(),
                    "message": "No text provided for depression detection"
                }
                return result
         
            # Preprocess text 
            processed_text = self.preprocess_text(text)

            # Tokenize
            max_length = self.CONFIG.DEPRESSION_MAX_LENGTH if hasattr(self.CONFIG, 'DEPRESSION_MAX_LENGTH') else 128
            
            encoding = self.TOKENIZER(
                processed_text,
                add_special_tokens=True,
                max_length=max_length,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt'
            )

            # Move to device
            input_ids = encoding['input_ids'].to(self.DEVICE)
            attention_mask = encoding['attention_mask'].to(self.DEVICE)

            # Predict
            with torch.no_grad():
                outputs = self.MODEL(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=1)
                pred_class = torch.argmax(probs, dim=1).item()
                confidence = probs[0][pred_class].item()

            # Get probabilities
            probability_not_depressed = probs[0][0].item()
            probability_depressed = probs[0][1].item()

            result = {
                "success": True,
                "timestamp": datetime.now().isoformat(),
                "prediction": "Depressed" if pred_class == 1 else "Not Depressed",
                "prediction_label": pred_class,
                "probability_not_depressed": float(f"{probability_not_depressed:.4f}"),
                "probability_depressed": float(f"{probability_depressed:.4f}"),
                "confidence": float(f"{confidence:.4f}"),
                "model_info": {
                    "name": self.CONFIG.DEPRESSION_MODEL_NAME,
                    "type": "BERT",
                    "device": str(self.DEVICE),
                    "max_length": max_length
                }
            }

            if save_result or self.CONFIG.DEPRESSION_SAVE_RESULTS:
                self.save_results_to_json(result)

            self.LOGGER.info(f"✓ Depression prediction complete: {result['prediction']}")
            return result
        
        except Exception as e:
            self.LOGGER.error(f"✗ Error during depression prediction: {str(e)}", exc_info=True)
            return {
                "success": False,
                "timestamp": datetime.now().isoformat(),
                "message": f"Error during processing: {str(e)}"
            }

    def save_results_to_json(self, result):
        """Save prediction results to JSON file"""
        os.makedirs(self.CONFIG.DEPRESSION_RESULTS_DIR, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"depression_result_{timestamp}.json"
        filepath = os.path.join(self.CONFIG.DEPRESSION_RESULTS_DIR, filename)
        with open(filepath, 'w') as f:
            json.dump(result, f, indent=2)

        self.LOGGER.info(f"Results saved to: {filepath}")

    def get_model_info(self):
        """Get model information"""
        return {
            "model_loaded": self.MODEL is not None,
            "tokenizer_loaded": self.TOKENIZER is not None,
            "model_name": self.CONFIG.DEPRESSION_MODEL_NAME,
            "model_type": "BERT (Hugging Face)",
            "device": str(self.DEVICE),
            "vocab_size": len(self.TOKENIZER.get_vocab()) if self.TOKENIZER else None
        }

    def health_check(self):
        """Check pipeline health"""
        return {
            "status": "healthy" if (self.MODEL is not None and self.TOKENIZER is not None) else "unhealthy",
            "model_loaded": self.MODEL is not None,
            "tokenizer_loaded": self.TOKENIZER is not None,
            "device": str(self.DEVICE),
            "config_loaded": self.CONFIG is not None,
            "model_info": self.get_model_info() if self.CONFIG else None
        }