import pickle
from sklearn.feature_extraction.text import CountVectorizer
import plotly.express as px
import pandas as pd
import re

def load_weights():
    cEXT = pickle.load( open( "./static/upload/models/cEXT.p", "rb"))
    cNEU = pickle.load( open( "./static/upload/models/cNEU.p", "rb"))
    cAGR = pickle.load( open( "./static/upload/models/cAGR.p", "rb"))
    cCON = pickle.load( open( "./static/upload/models/cCON.p", "rb"))
    cOPN = pickle.load( open( "./static/upload/models/cOPN.p", "rb"))
    vectorizer_31 = pickle.load( open( "./static/upload/models/vectorizer_31.p", "rb"))
    vectorizer_30 = pickle.load( open( "./static/upload/models/vectorizer_30.p", "rb"))
    
    return cEXT,cNEU,cAGR,cCON,cOPN,vectorizer_31,vectorizer_30

def predict_personality(cEXT,cNEU,cAGR,cCON,cOPN,vectorizer_31,vectorizer_30,text):
    scentences = re.split("(?<=[.!?]) +", text)
    text_vector_31 = vectorizer_31.transform(scentences)
    text_vector_30 = vectorizer_30.transform(scentences)
    EXT = cEXT.predict(text_vector_31)
    NEU = cNEU.predict(text_vector_30)
    AGR = cAGR.predict(text_vector_31)
    CON = cCON.predict(text_vector_31)
    OPN = cOPN.predict(text_vector_31)
    return [EXT[0], NEU[0], AGR[0], CON[0], OPN[0]]