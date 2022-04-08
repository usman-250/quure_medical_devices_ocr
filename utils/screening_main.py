from utils import personality_helper
from utils import scoring_helpers
from utils import helpers
from spacy import displacy

import visualise_spacy_tree
import numpy as np
import spacy
import glob
import json
import sys


# load english language model
nlp = spacy.load('en_core_web_sm',disable=['ner','textcat'])


# job_description = 'Python, data science, data analysis, deep learning, machine learning, pandas, numpy, sklearn' # arg parser
def main_fun(job_description,jd_select_box):
    try:
        json_path = './static/upload/linkedin_json/result.json'
        resumes_path = './static/upload/resumes/*.*'
        tree_sheet_path = './static/upload/tree_sheet/Tree.xlsx'
        print("profiles_info")
        profiles_info = scoring_helpers.screening_main( json_path, job_description, resumes_path, tree_sheet_path) 
        print("profiles_info")
        data = scoring_helpers.read_json(json_path)
        print("profiles_info")
        cEXT, cNEU, cAGR, cCON, cOPN, vectorizer_31, vectorizer_30 = personality_helper.load_weights()
        for key in data.keys():
            try:
                temp_dict = {}
                temp_list = []
                df = helpers.make_df(data[key])
                df['Speech_clean'] = df['Speech'].apply(helpers.clean)
                df['sent'] = df['Speech_clean'].apply(helpers.sentences)
                df2 = helpers.make_sentences_df(df)
                df2['Schemes1'] = df2.apply(lambda x:helpers.all_schemes(x.Sent,nlp,jd_select_box),axis=1)
                count = 0
                for i in range(len(df2)):
                    if len(df2.loc[i,'Schemes1'])!=0:
                        temp_list.append(df2.loc[i,'Sent'])
                        count+=1
                if temp_list:
                    if count/len(data[key]['activity']) > 0.99:
                        profiles_info[key]['activity_similar'] = 0.99
                    else:

                        profiles_info[key]['activity_similar'] = count/len(data[key]['activity'])
                else:
                    count = 0
                    profiles_info[key]['activity_similar'] = count

                if 'about' in data[key].keys():
                    predictions = personality_helper.predict_personality(cEXT, cNEU, cAGR, cCON, cOPN, vectorizer_31, vectorizer_30,data[key]['about'])
                else:
                    predictions = personality_helper.predict_personality(cEXT, cNEU, cAGR, cCON, cOPN, vectorizer_31, vectorizer_30,data[key]['name'])

                profiles_info[key]['personality'] = [int(i) for i in predictions]

            except:
                print(key)
                profiles_info[key]['activity_similar'] = 0
                profiles_info[key]['personality'] = []
                pass
        with open('../auto_recruitment/auto_recruitment/dashboard_json/'+jd_select_box+'.json', 'w') as fp:
            json.dump(profiles_info, fp)        
        return True
    except Exception as e:
        print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)
        return False
