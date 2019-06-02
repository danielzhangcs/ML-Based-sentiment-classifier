import os
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import lexicon_reader
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
from nltk.corpus import stopwords
import re

privative_list=["don't", "can't", "won't","isn't", "doesn't","couldn't","hasn't","didn't","weren't","aren't","wasn't", "wouldn't","haven't","Shouldn't","no","not","never","none",
                "nobody","nothing","nowhere","neither","hardly","scarcely","barely","little","few","seldom"]

degree_adverb = ["absolutely", "altogether", "completely", "entirely", "extremely", "fully", "perfectly", "quite", "thoroughly", "totally", "utterly", "wholly","badly", "bitterly",
                 "deeply", "enormously", "far", "greatly", "largely", "particularly", "profoundly", "so", "strongly", "terribly", "tremendously", "vastly", ]




class CustomFeats(BaseEstimator, TransformerMixin):
    """Extract features from each document for DictVectorizer"""
    def __init__(self, filedir):
        self.feat_names = set()
        lexicon_dir = os.path.join(filedir, 'lexicon')
        self.inqtabs_dict = lexicon_reader.read_inqtabs(os.path.join(lexicon_dir, 'inqtabs.txt'))
        self.swn_dict = lexicon_reader.read_senti_word_net(os.path.join(lexicon_dir, 'SentiWordNet_3.0.0_20130122.txt'))

    def fit(self, x, y=None):
        return self

    @staticmethod
    def word_count(review):
        words = review.split(' ')
        return len(words)

    def pos_count(self, review):
        words = review.split(' ')
        count = 0
        for word in words:
            if word in self.inqtabs_dict.keys() and self.inqtabs_dict[word] == lexicon_reader.POS_LABEL:
                count += 1
        return count
    
    def neg_count(self, review):
        words = review.split(' ')
        count = 0
        for word in words:
            if word in self.inqtabs_dict.keys() and self.inqtabs_dict[word] == lexicon_reader.NEG_LABEL:
                count += 1
        return count


    

    def features(self, review):
        return {
            'length': len(review),
            'num_sentences': review.count('.'),
            'num_words': self.word_count(review),
             # 4 example features; add your own here e.g. word_count, and pos_count
            'pos':  self.pos_count(review),
            'neg':  self.neg_count(review),
            'good': review.find('good') >=0,
            'best':review.find('best') >=0,
            'BEST':review.find('BEST') >=0,
            'Thanks':review.find('Thanks') >=0,
            'a good':review.find('a good') >=0,
            'good movie':review.find('good movie') >=0,
            'very good':review.find('very good') >=0,
            'loved':review.find('loved') >=0,
            'great':review.find('great') >=0,
            'GREAT':review.find('GREAT') >=0,
            'fans':review.find('fans') >=0,
            'lol':review.find('lol') >=0,
            'outstanding':review.find('outstanding') >=0,
            'cool':review.find('cool') >=0,
            'like':review.find('like') >=0,
            'love':review.find('love') >=0,

            'rate_8':review.find('8') >=0,
            'rate_9':review.find('9') >=0,
            'rate_10':review.find('10') >=0,
            'enjoy':review.find('enjoy') >=0,
            'favorite':review.find('favorite') >=0,
            'recommend':review.find('recommand') >=0,
            'rate_1':review.find('1') >=0,
            'rate_2':review.find('2') >=0,
            'rate_3':review.find('3') >=0,
            'bad':review.find('bad') >=0,
            'awful':review.find('awful') >=0,
            'terrible':review.find('terrible') >=0,
            'a bad':review.find('a bad') >=0,
            'bad movie':review.find('bad movie') >=0,
            'weep':review.find('weep') >=0,
            '!>=3':len(list(filter(lambda r: r.find("!") is not -1, review.split(' '))))>=3,
            'good>=3':len(list(filter(lambda r: r.find("good") is not -1, review.split(' '))))>=3,
            'great>=3':len(list(filter(lambda r: r.find("great") is not -1, review.split(' '))))>=3,
            'like>=3':len(list(filter(lambda r: r.find("like") is not -1, review.split(' '))))>=3,
            'love>=3':len(list(filter(lambda r: r.find("great") is not -1, review.split(' '))))>=3
            


            
            
            
            
            
            
        }

    def get_feature_names(self):
        return list(self.feat_names)

    def transform(self, reviews):
        feats = []
        for review in reviews:
            f = self.features(review)
            [self.feat_names.add(k) for k in f]
            feats.append(f)
        return feats


def get_custom_vectorizer():
    # Experiment with different vectorizers
    return CountVectorizer()
