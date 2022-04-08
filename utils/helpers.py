import re
from spacy.matcher import Matcher 
import pandas as pd
import nltk
def make_df(data):
    df = pd.DataFrame(columns={'time','Speech','actor_name'})
    i = 0 
    for items in data['activity']:
        df.loc[i,'time'] = items['time']
        df.loc[i,'Speech'] = items['text']
        df.loc[i,'actor_name'] = items['actor_name']
        i+=1
    return df

def make_sentences_df(df):
    # Create a dataframe containing sentences
    df2 = pd.DataFrame(columns=['Sent','time','Len'])

    # List of sentences for new df
    row_list = []

    # for-loop to go over the df speeches
    for i in range(len(df)):

        # for-loop to go over the sentences in the speech
        for sent in df.loc[i,'sent']:

            wordcount = len(sent.split())  # Word count
            year = df.loc[i,'time']  # Year
            dict1 = {'time':year,'Sent':sent,'Len':wordcount}  # Dictionary
            row_list.append(dict1)  # Append dictionary to list
    # Create the new df
    return pd.DataFrame(row_list)


# function to preprocess speech
def clean(text):
    
    # removing paragraph numbers
    text = text.replace('#','')
    text = re.sub('[0-9]+.\t','',str(text))
    # removing new line characters
    text = re.sub('\n ','',str(text))
    # removing apostrophes
    text = re.sub("'s",'',str(text))
    # removing hyphens
    text = re.sub("-",' ',str(text))
    text = re.sub("— ",'',str(text))
    # removing quotation marks
    text = re.sub('\"','',str(text))
    # removing salutations
    text = re.sub("Mr\.",'Mr',str(text))
    text = re.sub("Mrs\.",'Mrs',str(text))
    # removing any reference to outside text
    text = re.sub("[\(\[].*?[\)\]]", "", str(text))
    
    return text

# split sentences
def sentences(text):
    # split sentences and questions
    text = re.split('[.?]', text)
    clean_sent = []
    for sent in text:
        clean_sent.append(sent)
    return clean_sent

# To extract initiatives using pattern matching
def all_schemes(text,nlp, jd ):
    
    schemes=[]
    
    # Initiatives keywords
    data_science = [' AI ','scheme',
                 'learning','machine',
                 'data science',' nlp ',
                 'model',' ML ']
    
    web_development = ['web','development',
                 'back end','software',
                 'software','development',
                 'backend','Python',
                 'Javascript','Django']
    
    data_engineering = ['data', 'analysis',
                    'data', 'engineering',
                    'data', 'processing']
    
    h_r = ['Employee', 'Relations',
            'Employee','Engagement',
            'communication','skills',
            'Job','Evaluation',
            'organizational','skills']
    
    prog_list = {
                 'data science':data_science,
                 'web development':web_development,
                 'data engineering':data_engineering,
                 'hr' : h_r,
                }
    
    bigrams_trigrams = list(map(' '.join, nltk.everygrams(prog_list[jd.lower()], 1,2 )))
    for i in bigrams_trigrams:
        if i.lower() in text.lower():
            schemes.append(text)
        
    return schemes



