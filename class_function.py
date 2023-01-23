import pandas as pd
import numpy as np
import time
import math
from collections import Counter
from nltk.metrics import edit_distance
from textblob import TextBlob, Word
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from langdetect import detect
import os
import joblib
import pickle
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import PorterStemmer
import requests
from bs4 import BeautifulSoup
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt


class review_feature:
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()
        self.hindi_swear_words = None
        self.english_swear_words = None
        self.tagger = None
        self.company_tag = None
        self.model_data = None
        pass

    def hindi_swear_check(self,string, print_word = False):
        '''
        input: string
        output: True if text has hinglish proganity False if no profanity
        '''
        if self.hindi_swear_words == None:
            self.hindi_swear_words = []
            try:
                with open('utils/DictionaryUtils/hindi_swear_words.txt','r') as fp:
                    data = fp.read().lower()
                result = data.split('\n')
                self.hindi_swear_words = set([x.split('~')[0][:-1].lower() for x in result])
            except:
                print('Warning: hindi_swear_words.txt not read')
                pass
            self.hindi_swear_words = set(self.hindi_swear_words)
            if '' in self.hindi_swear_words or ' ' in self.hindi_swear_words:
                self.hindi_swear_words.pop()

        for word in self.hindi_swear_words:
            if word in string.lower().split():
                if print_word == True:
                    print(word)
                return True
        return False

    def english_swear_check(self,string, print_word = False):
        '''
        input: string
        output: True if text has english proganity False if no profanity
        '''
        if self.english_swear_words == None:
            self.english_swear_words = []
            try:
                with open('utils/DictionaryUtils/english_profanity_google.txt','r') as fp:
                    data = fp.read().lower()
                self.english_swear_words = set(data.split('\n'))
            except:
                print('Warning: english_profanity_google.txt not read')
                pass
            self.english_swear_words = set(self.english_swear_words)
            if '' in self.english_swear_words or ' ' in self.english_swear_words:
                self.english_swear_words.pop()

        for word in self.english_swear_words:
            if word in string.lower().split():
                if print_word == True:
                    print(word)
                return True
        return False

    def spell_correct(self,text,spell_threshold):
        '''
        text: string input
        spell_threshold: how much correction is required | keeping value higher is better
        '''
        text_list = text.split()
        ouput = " "
        for i in range(len(text_list)):
            w = Word(text_list[i])
            if w.spellcheck()[0][1]>spell_threshold:
                text_list[i] = w.spellcheck()[0][0]
        return ouput.join(text_list)

    def service_tag(self,text, print_word = False):
        '''
        text: string input
        output: 0 or 1
        '''
        if self.tagger == None:
            self.tagger = []
            try:
                with open('utils/DictionaryUtils/service_tagger.txt','r') as fp:
                    data = fp.read().lower()
                self.tagger = set(data.split('\n'))
            except:
                print('Warning: Service_tagger.txt not read')
                pass
            self.tagger = set(self.tagger)

            if '' in self.tagger or ' ' in self.tagger:
                self.tagger.pop()

        k = text.split()
        for w in k:
            for wrd in self.tagger:
                x = edit_distance(w.lower(),wrd)
                if x<=1:
                    if print_word == True:
                        print(wrd)
                    return 1
        return 0

    def polarity_sentiment(self,text):
        '''
        input: string
        output: value between -1 to +1
        '''
        blob = TextBlob(text)
        return (blob.sentiment.polarity)

    def subjectivity_sentiment(self,text):
        '''
        input: string
        output: 0 to 1
        '''
        blob = TextBlob(text)
        return (blob.sentiment.subjectivity)

    def slang_emoji_polarity_compoundscore(self,text):
        '''
        Input: Text
        Output:
        (-0.5 to +0.5): Neural
        (-inf to -0.5): Negative
        (+0.5 to +inf): Positive
        '''
        return self.analyzer.polarity_scores(text)['compound']

    def string_comparison(self,text1,text2,choice='levenshtein_distance'):
        '''
        text1: String Input 1
        text2: String Input 2
        choice: 'levenshtein_distance' or 'damerau_levenshtein_distance' or 'hamming_distance' or 'jaro_distance' or 'jaro_winkler' or 'match_rating_comparison'
        '''
        # https://jellyfish.readthedocs.io/en/latest/comparison.html
        if choice == 'levenshtein_distance':
            return jellyfish.levenshtein_distance(text1,text2)
        elif choice == 'damerau_levenshtein_distance':
            return jellyfish.damerau_levenshtein_distance(text1,text2)
        elif choice == 'hamming_distance':
            return jellyfish.hamming_distance(text1,text2)
        elif choice == 'jaro_distance':
            return jellyfish.jaro_distance(text1,text2)
        elif choice == 'jaro_winkler':
            return jellyfish.jaro_winkler(text1,text2)
        elif choice == 'match_rating_comparison':
            return jellyfish.match_rating_comparison(text1,text2)
        else:
            print("Wrong Choice")

    def gibberish_detection(self,l, prefix_path = './'):
        '''
        Input: String
        prefix_path: path of gibberish pickle weights
        Output: True or False
        '''
        if self.model_data == None:
            self.model_data = pickle.load(open('utils/DictionaryUtils/gib_model.pki', 'rb'))

        accepted_chars = 'abcdefghijklmnopqrstuvwxyz '
        pos = dict([(char, idx) for idx, char in enumerate(accepted_chars)])

        def normalize(line):
            """ Return only the subset of chars from accepted_chars.
            This helps keep the  model relatively small by ignoring punctuation,
            infrequenty symbols, etc. """
            return [c.lower() for c in line if c.lower() in accepted_chars]

        def ngram(n, l):
            """ Return all n grams from l after normalizing """
            filtered = normalize(l)
            for start in range(0, len(filtered) - n + 1):
                yield ''.join(filtered[start:start + n])

        def avg_transition_prob(l, log_prob_mat):
            """ Return the average transition prob from l through log_prob_mat. """
            log_prob = 0.0
            transition_ct = 0
            for a, b in ngram(2, l):
                log_prob += log_prob_mat[pos[a]][pos[b]]
                transition_ct += 1
            # The exponentiation translates from log probs to probs.
            return math.exp(log_prob / (transition_ct or 1))

        model_mat = self.model_data['mat']
        threshold = self.model_data['thresh']
        return (avg_transition_prob(l, model_mat) < threshold)

    def language_detection(self, text):
        '''
        :param text: Text for which to detect language
        :return: `hi` or `bi` or `en`, etc
        Source: https://github.com/Mimino666/langdetect
        '''
        return detect(text)

    def competitive_brand_tag(self, text, word_distance = 1, print_word = False):
        '''
        :param text: input review string
        :param word_distance: word distance b/w review word and company word (amazon, amzon): helps avoid spell error
        :param print_word: print which company tag is matching
        :return: True (company tag present in review) or False (company tag not present in review)
        '''
        if self.company_tag is None:
            self.company_tag = []
            with open('utils/DictionaryUtils/company_tags.txt','r') as fp:
                data = fp.read().lower()
            self.company_tag = data.split('\n')
            self.company_tag = set(self.company_tag)
            # print(self.company_tag)

        input_str = text.split()
        for x in input_str:
            for y in self.company_tag:
                try:
                    if self.string_comparison(text1=x, text2=y, choice='damerau_levenshtein_distance') <= word_distance:
                        if print_word:
                            print("Delete for:", x, y)
                        return True
                except:
                    pass
        return False

    def corpus_stem_lemma(self, corpus):
        '''
        Input: Corpus(List of Strings)
        Output: A lemmatized and stemmed Corpus
        '''
        for i in range(len(corpus)):
            doc = nlp(corpus[i])
            corpus[i] = " ".join([stemmer.stem(token.lemma_) for token in doc if token.is_stop == False and token.is_punct == False and token.is_alpha == True])
            #print(temp[i])
        return corpus

    def noun_score(self, corpus):
        '''
        TFIDF_NOUN_SCORE = Sum of TFIDF OF NOUN in a Review / Sum of TFIDF of all words in that review
        :param corpus:
        :return:
        '''
        noun_tag = []
        for review in corpus:
            doc = nlp(review)
            noun_tag.append([stemmer.stem(token.lemma_) for token in doc if token.pos_ == "NOUN" and token.is_stop == False and token.is_punct == False and token.is_alpha == True])

        corpus = self.corpus_stem_lemma(corpus)

        tfidf = TfidfVectorizer(min_df=1, ngram_range=(1, 1))

        features = tfidf.fit_transform(corpus)
        df_tfidf = pd.DataFrame(features.todense(), columns=tfidf.get_feature_names())
        df_tfidf['sum'] = df_tfidf.sum(axis=1)

        df_tfidf['noun_sum'] = 0.0
        df_tfidf['tfidf_score'] = 0.0

        for i in range(len(noun_tag)):
            sm = 0.0
            for q in noun_tag[i]:
                if q in df_tfidf.columns:
                    sm += df_tfidf[q][i]
            df_tfidf.at[i, 'noun_sum'] = sm
            if df_tfidf.at[i, 'sum'] == 0.0:
                df_tfidf.at[i, 'tfidf_score'] = 0.0
                continue
            df_tfidf.at[i, 'tfidf_score'] = float(df_tfidf.at[i, 'noun_sum'] / df_tfidf.at[i, 'sum'])

        return df_tfidf['tfidf_score']


nlp = spacy.load("en_core_web_sm")
stemmer = PorterStemmer()

rf = review_feature()


def data_preprocess(df):
    df['review_len'] = df['answer_option'].apply(lambda x: len(x.split()))
    
    #exception for slang
    bad_reviews = []
    for indx in df.index:
        review = df.at[indx, 'answer_option']
        try:
            b = rf.language_detection(review)
            if b == 'hi' or b == 'mr':
                bad_reviews.append(indx)
        except:
            bad_reviews.append(indx)
    df = df[~df.index.isin(bad_reviews)].reset_index(drop = True)
    
    #Gibberish
    bad_reviews = []
    for indx in df.index:
        review = df.at[indx, 'answer_option']
        if rf.gibberish_detection(review,prefix_path='utils' ):
            bad_reviews.append(indx)
    df = df[~df.index.isin(bad_reviews)].reset_index(drop = True)
    
    #Profanity Detection
    bad_reviews = []
    for indx in df.index:
        review = df.at[indx, 'answer_option']
        if rf.english_swear_check(review) or rf.hindi_swear_check(review):
            bad_reviews.append(indx)
    df = df[~df.index.isin(bad_reviews)].reset_index(drop = True)
    
    #Spelling Correction
    for indx in df.index:
        review = df.at[indx, 'answer_option']
        df.at[indx, 'answer_option'] = rf.spell_correct(review, 0.9)
        
    #Company Tag
    bad_reviews = []
    for indx in df.index:
        review = df.at[indx, 'answer_option']
        if rf.competitive_brand_tag(review):
            bad_reviews.append(indx)
    df = df[~df.index.isin(bad_reviews)].reset_index(drop = True)
    
    return df

def feature_eng(df):
    df = df.sort_values(by = ['product'], axis = 0)
    
    #Add this features into the columns
    df['Rn'] = 0.0
    df['Rp'] = 0.0
    df['Rs'] = 0.0
    df['Rc'] = 0.0
    df['Rd'] = 0.0
    df['Rsc'] = 0.0
    
    product_list = df['product'].unique()
    
    for product in product_list:
        data = df[df['product']==product]
        unique_bag = set()
        for review in data['answer_option']:
            review = review.lower()
            words = review.split()
            unique_bag = unique_bag.union(set(words))

        for indx in data.index:
            review = data.at[indx, 'answer_option']
            df.at[indx, 'Rp'] = rf.polarity_sentiment(review)
            df.at[indx, 'Rs'] = rf.subjectivity_sentiment(review)
            df.at[indx, 'Rd'] = rf.service_tag(review)
            df.at[indx, 'Rsc'] = rf.slang_emoji_polarity_compoundscore(review)
            df.at[indx, 'Rc'] = float(len(set(review.split()))) / float(len(unique_bag))

    df.loc[df['product']==product, 'Rn'] = rf.noun_score(data['answer_option'].values).values
    
    df.to_csv('process/Feature_Eng_Reviews.csv', index=None)

classifier = joblib.load('model/randomforest.joblib')

def result_gen(df):
    product_list = df['product'].unique()
    df['win']=0
    df['lose']=0
    df['review_score'] = 0.0
    df.reset_index(inplace = True, drop = True)
    
    def score_giver(C,D):
        E = pd.merge(C,D,how='outer',on='j')
        E.drop(columns=['j'],inplace = True)
        q= classifier.predict(E.values)
        return Counter(q)

    for product in product_list:
        data = df[df['product']==product]
        for indx in data.index:
            review = df.iloc[indx, 2:-3]
            review['j'] = 'jn'
            C = pd.DataFrame([review])
            D = data[data.index!=indx].iloc[:,2:-3]
            D['j'] = 'jn'
            score = score_giver(C,D)
            df.at[indx, 'win'] = 0 if score.get(1) is None else score.get(1)
            df.at[indx, 'lose'] = 0 if score.get(0) is None else score.get(0)
            df.at[indx, 'review_score'] = float(0 if score.get(1) is None else score.get(1)) / len(data) * 1.0
    
    df = df.sort_values(by = ['product','review_score'], ascending = False)
    
    df.iloc[:, [0,1,-1]].to_csv('process/data_ranked_output.csv',index = False)

def final_result(df):
    data_pre = data_preprocess(df)
    feature_eng(data_pre)
    
    feature_df = pd.read_csv('process/Feature_Eng_Reviews.csv')
    
    result_gen(feature_df)
    
    result_final = pd.read_csv('process/data_ranked_output.csv')
    
    return result_final

def scrape_review_in_list(url,HEADERS):
    review = []
    
    data = requests.get(url,headers=HEADERS)
    page_soup=BeautifulSoup(data.text,'html.parser')
    parse_link = page_soup.find_all("a", class_="a-link-emphasis a-text-bold")[0]['href'][1:]
    main_page_link = "https://www.amazon.in/"+ parse_link
    url = main_page_link.split('UTF8')[0] + "UTF8&pageNumber=1&reviewerType=all_reviews"
    i=1
    while i!=0:
        data = requests.get(url,headers=HEADERS)
        soup = BeautifulSoup(data.text, 'html.parser')
        review_10 = soup.find_all("span", class_="a-size-base review-text review-text-content")
        if len(review_10)==10:
            for j in review_10:
                    rev_result = j.get_text()
                    rev_result = rev_result[1:]
                    rev_result = rev_result[:-1]
                    review.append(rev_result)
                    
            pageNumberOld = f'pageNumber={i}'
            pageNumber = f'pageNumber={i+1}'
            i=i+1
            url = url.replace(pageNumberOld,pageNumber)
        else:
            i=0
        
    return review

def link_to_result(url,prod_type):
    
    #header to scrape
    HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.3; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.71 Safari/537.36'}
    
    x = scrape_review_in_list(url,HEADERS)
    
    req_df_dict = {'product': [prod_type]*len(x),
              'answer_option':x,}
    
    main_data_df_for_predict = pd.DataFrame(req_df_dict)
    
    main_data_df_for_predict.to_csv('process/Input.csv',index=None)
    
    result_new = final_result(main_data_df_for_predict)
    return result_new


# uri = "https://www.amazon.in/LG-Convertible-Anti-Virus-Protection-RS-Q19YNZE/dp/B0BQ3MXML8/ref=cm_cr_arp_d_product_top?ie=UTF8"
# res_comp = link_to_result(uri,"earphone")


def wordcloud_gen(df):
    comment_words = ''
    stopwords = set(STOPWORDS)
    
    # iterate through the csv file
    for val in df.answer_option:
        
        # typecaste each val to string
        val = str(val)
    
        # split the value
        tokens = val.split()
        
        # Converts each token into lowercase
        for i in range(len(tokens)):
            tokens[i] = tokens[i].lower()
        
        comment_words += " ".join(tokens)+" "

    wc = WordCloud(width=1400,height=800,stopwords=stopwords,background_color='white').generate(comment_words)

    plt.figure(figsize=(20,10), facecolor='k', edgecolor='k')
    plt.imshow(wc, interpolation='bicubic') 
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('static/images/woc.png')
    plt.close()