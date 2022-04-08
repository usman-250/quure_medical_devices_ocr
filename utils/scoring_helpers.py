from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from utils import tf_idf_cosine_similarity as tf_idf

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import nltk.corpus
import datetime
import textract
import spacy
import nltk
import glob
import json
import time
import re

stop_words_ = stopwords.words('english')
nlp = spacy.load('en_core_web_sm', disable=['ner', 'textcat'])
def extract_skills(input_text, skill_list):
    try:
        skill_list = [i.lower() for i in skill_list]

        input_text = input_text.lower()
        stop_words = set(nltk.corpus.stopwords.words('english'))
        word_tokens = nltk.tokenize.word_tokenize(input_text)

        # remove the stop words

        filtered_tokens = [w for w in word_tokens if w
                           not in stop_words]

        # remove the punctuation

        filtered_tokens = [w for w in word_tokens if w.isalpha()]

        # generate bigrams and trigrams (such as artificial intelligence)

        bigrams_trigrams = list(map(' '.join,
                                nltk.everygrams(filtered_tokens, 2, 3)))

        # we create a set to keep the results in.

        found_skills = list()

        # we search for each token in our skills database

        for token in filtered_tokens:
            if token.lower() in skill_list:
                found_skills.append(token)

        # we search for each bigram and trigram in our skills database

        for ngram in bigrams_trigrams:
            if ngram.lower() in skill_list:
                found_skills.append(ngram)

        return found_skills
    except Exception as e:
        print ('''exception in scoring_helpers.extract_skills fun ''', e)
        return []


def extract_text_from_document(filename):
    try:
        text = textract.process(filename)
        lower_case_string = str(text.decode('utf-8')).lower()
        return lower_case_string
    except Exception as e:
        print ('''exception in scoring_helpers.extract_text_from_document fun ''', e)
        return ''


def get_experience_description(experience_list):
    """
        input : list of all experiences
        output (str) : return description of all experiences
        working : get description of all experiences mentioned in linkedin
    """

    my_exp = ''
    try:
        for exp in experience_list:
            if 'Description' in exp.keys():
                my_exp += ' ' + exp['Description'].lower()
            if 'sub_categories' in exp.keys():  # ['sub_categories'][0]['Description']
                for sub in exp['sub_categories']:
                    if 'Description' in sub.keys():
                        my_exp += ' ' + sub['Description']
    except Exception as e:
        print ('''exception in scoring_helpers.get_experience_description fun ''', e)
        pass
    return my_exp


def get_total_experience_and_company(experience_list):
    """
        input : list of all experiences
        output (dict) : return total experience and experience in each organization 
        working : get total experience and experience in each organization
    """

    company_exp = {}
    total_exp = 0
    total_years = 0
    total_months = 0
    try:
        for exp in experience_list:
            if 'Total Duration' in exp.keys():
                (years, months) = (0, 0)
                try:
                    my_exp = exp['Total Duration'].replace('s', '').strip()
                    if 'yr' in my_exp:
                        if len(my_exp.split('yr')) > 1 and '' not in my_exp.split('yr'):
                            years = int(my_exp.split('yr')[0].strip())
                            months = int(my_exp.split('yr')[1].strip(' mos'))
                        else:
                            years = int(my_exp.split('yr')[0].strip())
                    else:
                        months = int(my_exp.strip(' mos'))
                except:
                    months = 0
                    years = 0

                total_years += years
                total_months += months

            if 'Company Name' in exp.keys():
                if 'less than a year' not in exp['Total Duration']:
                    company_exp[exp['Company Name'].split('Full-time')[0].strip()] = exp['Total Duration']

        total_exp = total_years + round(total_months / 12, 1)
        return (company_exp, total_exp)
    except Exception as e:
        print ('''exception in scoring_helpers.get_total_experience_and_company fun ''', e)
        return (company_exp, total_exp)


def all_employments(exp_lst):
    """
        input : list of all experiences
        output (list) : return duration of each employment and employment year 
        working : get duration of each employment and employment year 
    """

    employment_lst = []
    durations = []
    try:
        for exp in exp_lst:
            if 'Employment' in list(exp.keys()):
                employment_lst.append(exp['Employment'])
            elif 'sub_categories' in list(exp.keys()):

                sub_categories = exp['sub_categories']
                for sub_cat in sub_categories:
                    if 'Employment' in list(sub_cat.keys()):
                        employment_lst.append(sub_cat['Employment'])

            if 'Total Duration' in list(exp.keys()):
                durations.append(exp['Total Duration'])
        return (employment_lst, durations)
    except Exception as e:
        print ('''exception in scoring_helpers.all_employments fun ''', e)
        return (employment_lst, duration)


def get_employment_period(employment):
    """ 
        input (str) : a string containing employment ( starts,ends )
        output (list) : list containing start and end of employment
        working : Function to extract start and end of employment
    """

    emp_from_to = []
    try:
        for i in employment.split():
            if i.isdigit() and len(i) == 4 or i == 'Present':
                emp_from_to.append(i)
    except Exception as e:
        print ('''exception in scoring_helpers.get_employment_period fun ''', e)
        pass
    return emp_from_to


def exp_to_from(employment_lst):
    """
        input (list) :  list of all employments
        output (list) : return list of start and end of employment e.g :  [(start, end), (start, end) ] 
    """

    exp_starts_ends = []
    try:
        for emp in employment_lst:
            employment_period = get_employment_period(emp)
            exp_starts_ends.append((employment_period[0],
                                   employment_period[1]))
    except Exception as e:
        print ('''exception in scoring_helpers.exp_to_from fun ''', e)
    return exp_starts_ends


def get_total_exp(exp_starts_ends):
    """
        input (tuple): list of tuples containing start and end of employment
        output ( int ): total years of experience 
    """

    try:
        exp_starts_ends = sorted(exp_starts_ends, key=lambda x: x[0])

        total_exp = 0
        present = False
        indexed = False

        for (ind, exp) in enumerate(exp_starts_ends):
            if indexed:
                indexed = False
                continue

            if exp[0] == '':
                continue

            job_starts = int(exp[0])
            if exp[1] != 'Present':
                job_ends = int(exp[1])
            else:
                year = str(datetime.datetime.now().year)
                job_ends = int(year)
                present = True

            if present:
                total_exp += job_ends - job_starts
                break
            else:
                if job_ends <= int(exp_starts_ends[ind + 1][0]):
                    total_exp += job_ends - job_starts
                else:
                    if exp_starts_ends[ind + 1][1] == 'Present':
                        next_job_ends = datetime.datetime.now().year
                        total_exp += next_job_ends - job_starts
                        break
                    else:
                        next_job_ends = int(exp_starts_ends[ind + 1][1])
                        if next_job_ends <= job_ends:
                            total_exp += job_ends - job_starts
                        else:
                            total_exp += next_job_ends - job_starts
                            indexed = True
        return total_exp
    except Exception as e:
        print ('''exception in scoring_helpers.get_total_exp fun ''',e)
        return 0


def read_json(json_path):
    try:

        # Opening JSON file

        f = open(json_path, 'r')

        # returns JSON object as
        # a dictionary

        data = json.load(f)
        return data
    except Exception as e:
        print ('''exception in scoring_helpers.read_json fun ''', e)
        return ''


def read_all_resumes(resumes_path):
    """
        input (str) : path to resumes dir
        output (list): return list of all the resumes text
    """

    all_resumes_text = []
    try:
        all_resumes = glob.glob(resumes_path)
        for resume in all_resumes:
            text = extract_text_from_document(resume)
            all_resumes_text.append(text)
    except Exception as e:
        print ('''exception in scoring_helpers.read_all_resumes fun ''', e)
        pass
    return all_resumes_text


def get_profile_resume(all_resumes, name, company_name):
    try:
        for resume in all_resumes:
            resume = resume.lower()
            if name.lower() in resume and company_name.lower() in resume:
                return resume
        return ''
    except Exception as e:
        print ('''exception in scoring_helpers.get_profile_resume fun ''', e)
        return ''


def read_excel_sheet(sheet_path, industry, sheet_name, header=0):
    """
        input (str) : path of excel sheet
        input (str) : name of industry
        input (str) : name of sheet
        output ( DataFrame ): return dataframe of the respective sheet
    """

    try:
        df_sheet = pd.read_excel(sheet_path, sheet_name=industry + '_' + sheet_name, header=header)
        return df_sheet
    except Exception as e:
        print ('''exception in scoring_helpers.read_excel_sheet fun ''', e)
        return None


def all_schemes(text, skill_list):
    """
        input (str) : string to extract skills
        input (list) : list of skills to be extracted from text
        output (list) : return list of all the skills extracted from text
    """

    schemes = []
    try:
        bigrams_trigrams = list(map(' '.join,
                                nltk.everygrams(skill_list, 1, 2)))
        for i in bigrams_trigrams:
            if i.lower() in text:
                schemes.append(i)
    except Exception as e:
        print ('''

exception in scoring_helpers.all_schemes fun ''', e)
        pass
    return schemes


def remove_stopwords_tokenization(my_text):
    """
        input (str) : string containing all the text
        output (list) : list of filtered text (tokeniztion, remove stop words)
    """

    try:
        stop_words_.append(',')
        stop_words_.append('?')
        split_words = my_text.split()
        pair_words_tokens = list(map(' '.join, zip(split_words[:-1],
                                 split_words[1:])))
        word_tokens = word_tokenize(my_text)
        word_tokens.extend(pair_words_tokens)
        filtered_sentence = [w for w in word_tokens if not w.lower()
                             in stop_words_]
        filtered_sentence = [each_string.lower().strip()
                             for each_string in filtered_sentence]
        return filtered_sentence
    except Exception as e:
        print ('''exception in scoring_helpers.remove_stopwords_tokenization fun ''', e)
        return []


def extract_skills_from_corpus(filtered_sentence, df):
    """
        input (list): list of filtered text after tokenization
        input ( dataframe ): dataframe containing skills
        output (dict) : return dict of skills and thier roles(domains)
    """

    skills_found = {}
    try:
        for column in df.columns:
            skill_list = df[column].to_list()
            skill_list = [each_string.lower().strip()
                          for each_string in skill_list]
            skill_list.append(column.lower())
            skill_list = list(filter(None, skill_list))
            schemes = all_schemes(filtered_sentence, skill_list)
            if schemes:
                skills_found[column] = schemes
    except Exception as e:
        print ('''exception in scoring_helpers.extract_skills_from_corpus fun ''', e)
        pass

#     return [item for sublist in skills_found for item in sublist]

    return skills_found


def get_common_skills_and_roles(roles_with_skills_and_aliases,
                                skills_roles):
    """

        output (dict) : return dict of respective domains and skills 
    """

    try:
        s1 = pd.DataFrame(roles_with_skills_and_aliases)
        s2 = pd.DataFrame(skills_roles)
        s1['Aliases'] = s1['Aliases'].astype(object)
        s2['Aliases'] = s2['Aliases'].astype(object)
        intersection_df = pd.merge(s1, s2)
        d = intersection_df.to_dict()
        
        if s1.empty:
            d['s1'] = 's1 is empty'
        if s2.empty:
            d['s2'] = 's2 is empty'

    #     intersection_df = drop_duplicates(intersection_df)

        return d
    except Exception as e:
        print ('''exception in scoring_helpers.get_common_skills_and_roles fun ''', e)
        return {}


def drop_duplicates(df):
    """
    input (DataFrame) : pandas dataframe
    output (dataframe): return dataframe after removing duplicate rows
    """

    try:
        df.fillna(np.nan, inplace=True)
        L = []
        for (index, row) in df.iterrows():
            L.append([row['Role'], row['Skills'], row['Aliases'],
                     row['Parent']])
        new_k = []
        for elem in L:
            if elem not in new_k:
                new_k.append(elem)
        L = new_k
        temp = []
        temp2 = []
        for elem in L:
            if elem != [elem[0], elem[1], np.nan, elem[3]]:
                temp.append(elem)
            else:
                temp2.append(elem)
        final1 = []
        for elem in temp2:
            decision = []
            for elems in temp:
                if [elem[0], elem[1], elem[3]] == [elems[0], elems[1], elems[3]]:
                    decision.append(1)
                else:
                    decision.append(0)
            if sum(decision) == 0:
                final1.append(elem)
        temp.extend(final1)
        dff = pd.DataFrame(temp, columns=['Role', 'Skills', 'Aliases', 'Parent'])

        return dff
    except Exception as e:
        print ('''exception in scoring_helpers.drop_duplicates fun ''', e)
        return None


def get_union_skills_and_roles(roles_with_skills_and_aliases,
                               skills_roles):
    """
    input (dict): dictionary contains roles as keys and skills as value
    input (dict): dictionary contains skills as keys and aliases as value
    
    output (dict): return dictionary (union of both input dicts)
    """
    try:
        s1 = pd.DataFrame(roles_with_skills_and_aliases)
        s2 = pd.DataFrame(skills_roles)

        unioun_df = pd.concat([s1,
                              s2]).drop_duplicates().reset_index(drop=True)
        unioun_df = drop_duplicates(unioun_df)
        return unioun_df.to_dict()
    except Exception as e:
        print ('''exception in scoring_helpers.get_union_skills_and_roles fun ''', e)
        return {}


def skills_and_domains(tree_sheet_path, industry, sheet_name, text):
    try:
        if isinstance(text, str):
            filtered_sentence = remove_stopwords_tokenization(text)
        else:
            filtered_sentence = text

        st = time.time()
        df = read_excel_sheet(tree_sheet_path, industry, sheet_name)
        df = df.T
        df.columns = df.iloc[0]
        df = df.fillna('')[1:]

        skills_and_domains_dict = \
            extract_skills_from_corpus(filtered_sentence, df)

        new_skills_and_domains_dict = {}
        for key in skills_and_domains_dict.keys():
            if key.lower() in skills_and_domains_dict[key]:
                skills_and_domains_dict[key].remove(key.lower())
            if skills_and_domains_dict[key]:
                new_skills_and_domains_dict[key] = \
                    skills_and_domains_dict[key]

        return (new_skills_and_domains_dict, df)
    except Exception as e:
        print ('''exception in scoring_helpers.skills_and_domains fun ''', e)
        return ({}, None)


def get_column_names(value, df):
    """
    input : a df cell value to find its column name
    input : dataframe 
    
    output (list) : return names of the columns containing that value in dataframe
    """
    
    try:
        series = df.apply(lambda row: row[row == value].index, axis=1)
        clms = []
        for i in series:
            try:
                clms.append(i[0])
            except:
                pass
        return clms
    except Exception as e:
        print ('''exception in scoring_helpers.get_column_names fun ''', e)
        return []


def get_roles_with_skills_and_aliases(roles, aliases, skills):
    """
    input : dictionary of roles/domains
    input : 
    
    output (dict) :
    """
    try:
        for alias in aliases:
            columns = get_column_names(alias, skills)
            for column in columns:
                if column in roles.keys():
                    attributes_list = roles[column]
                    attributes_list = [k.lower() for k in attributes_list]
                    if alias.lower() not in attributes_list:
                        roles[column].append(alias.lower())
        testing = pd.DataFrame(dict([(k, pd.Series(v)) for (k, v) in roles.items()])).T
        testing = testing.reset_index()
        testing = testing.rename(columns={'index': 'Role'})
        testing = testing.melt(id_vars=['Role'], value_name='Skills')
        testing['Aliases'] = ''
        testing = testing.drop('variable', 1)
        for alias in aliases:
            for (index, row) in testing.iterrows():
                if alias.lower() == row['Skills']:
                    if len(aliases[alias]) < 2 and row['Aliases'] == '':
                        row['Aliases'] = ''.join(aliases[alias])
                    elif len(aliases[alias]) >= 2 and row['Aliases'] == '':
                        for value in aliases[alias]:
                            if row['Aliases'] == '':
                                row['Aliases'] = value
                            else:
                                new_row = {'Role': row['Role'], 'Skills': row['Skills'], 'Aliases': value}
                                testing = testing.append(new_row, ignore_index=True)
        testing = testing.dropna()
        testing = testing.replace(r'^\s*$', np.nan, regex=True)
        testing['Parent'] = 'Skills'
        testing = testing.to_dict()
        return testing
    except Exception as e:
        print ('''exception in scoring_helpers.get_roles_with_skills_and_aliases fun ''', e)
        return {}

def skills_similarity_with_jd(job_description,skills_string):
    """
        Function will take job description and resumes (pdf or docs)
        vectorize documents using TF-IDF
        match vectors using cosine similarity
    """
    try:
        job_description = job_description #change here
        # print('The start' * 5)
        skills_text = [skills_string]

        # TF-IDF - cosine similarity
        cos_sim_list = tf_idf.get_tf_idf_cosine_similarity(job_description,skills_text)
    #     final_doc_rating_list = []
    #     zipped_docs = zip(cos_sim_list,resume_docs)
    #     sorted_doc_list = sorted(zipped_docs, key = lambda x: x[0], reverse=True)
    #     for element in sorted_doc_list:
    #         doc_rating_list = []
    #         doc_rating_list.append(os.path.basename(element[1]))
    #         doc_rating_list.append("{:.0%}".format(element[0]))
    #         final_doc_rating_list.append(doc_rating_list)
        return cos_sim_list
    except Exception as e:
        print ('''exception in scoring_helpers.skills_similarity_with_jd fun ''', e)
        return 0

def screening_main(json_path='', job_description='', resumes_path='', tree_sheet_path=''):
    """
        input (str) : path of json file
        input (str) : job description 
        output (dict): return a dictionary containing all info about profile
    """

    try:
        data = read_json(json_path)
    except:
        print('Json file path is incorrect.')
        data = {}

    profiles_info = {}

    # get all resumes text

    all_resumes_text = read_all_resumes(resumes_path)

    try:
        for profile in data.keys():
            my_profile = data[profile]
            my_name = my_profile['name']

            my_about = ''
            my_exp = ''
            my_skills = ''
            linkedin_skill_list = []
            my_exp_details = ''
            my_total_expp = 0

            if 'experience' in my_profile.keys():
                experience_list = my_profile['experience']

                # get total experience details

                (company_expp, t_exp) = \
                    get_total_experience_and_company(experience_list)
                (employment_lst, durations) = \
                    all_employments(experience_list)
                exp_starts_ends = exp_to_from(employment_lst)
                my_total_expp = get_total_exp(exp_starts_ends)
                my_exp = get_experience_description(experience_list)
                my_exp_details = (company_expp, my_total_expp)

            if 'skills' in my_profile.keys():
                for skill in my_profile['skills']:
                    my_skills += ' ' + skill.lower()
                    linkedin_skill_list.append(skill.lower())
            if 'about' in my_profile.keys():
                my_about += ' ' + my_profile['about'].lower()

            # ############## Industry Tree ###############

            # get working domains and skills from Linkedin

            (skills_aliases, aliases_df) = \
                skills_and_domains(tree_sheet_path, industry='IT',
                                   sheet_name='Aliases',
                                   text=linkedin_skill_list)
            (skills_roles, skills_df) = \
                skills_and_domains(tree_sheet_path, industry='IT',
                                   sheet_name='Skills',
                                   text=linkedin_skill_list)
            working_domains_and_skills_from_linkedin_skill_list = \
                get_roles_with_skills_and_aliases(skills_roles, 
                                                  skills_aliases, skills_df)

            # get working domains and skills from Experience

            (skills_aliases, aliases_df) = \
                skills_and_domains(tree_sheet_path, industry='IT',
                                   sheet_name='Aliases', text=my_exp)
            (skills_roles, skills_df) = \
                skills_and_domains(tree_sheet_path, industry='IT',
                                   sheet_name='Skills', text=my_exp)
            working_domains_and_skills_from_experience = \
                get_roles_with_skills_and_aliases(skills_roles,
                    skills_aliases, skills_df)

            try:

                # get profile's resume

                cmpny_name = \
                    list(company_expp.keys())[0].lower().replace('internship'
                        , '').strip()
                my_resume = get_profile_resume(all_resumes_text,
                        my_name, cmpny_name)
            except Exception as e:
                print ('Exception 2 maybe company not mentioned', e)
                my_resume = ''

            # get working domains and skills from reume

            (skills_aliases, aliases_df) = \
                skills_and_domains(tree_sheet_path, industry='IT',
                                   sheet_name='Aliases', text=my_resume)
            (skills_roles, skills_df) = \
                skills_and_domains(tree_sheet_path, industry='IT',
                                   sheet_name='Skills', text=my_resume)
            working_domains_and_skills_from_resume = \
                get_roles_with_skills_and_aliases(skills_roles,
                    skills_aliases, skills_df)

            # get working domains and skills from JD

            (skills_aliases, aliases_df) = \
                skills_and_domains(tree_sheet_path, industry='IT',
                                   sheet_name='Aliases',
                                   text=job_description)
            (skills_roles, skills_df) = \
                skills_and_domains(tree_sheet_path, industry='IT',
                                   sheet_name='Skills',
                                   text=job_description)
            working_domains_and_skills_from_JD = \
                get_roles_with_skills_and_aliases(skills_roles,
                    skills_aliases, skills_df)

            # get common working domains and skills from ( JD/Resume )

            common_resume_jd_skills = \
                get_common_skills_and_roles(working_domains_and_skills_from_resume,
                    working_domains_and_skills_from_JD)
            # get common working domains and skills from ( JD/Linkedin )

            common_linkedin_jd_skills = \
                get_common_skills_and_roles(working_domains_and_skills_from_experience,
                    working_domains_and_skills_from_JD)

            # get union of working domains and skills from ( linkedin/Resume )

            union_linkedin_resume_skills = \
                get_union_skills_and_roles(working_domains_and_skills_from_resume,
                    working_domains_and_skills_from_linkedin_skill_list)
            
            # get skills similarity with job description
            try:
                skills_jd = pd.DataFrame(working_domains_and_skills_from_JD)
                skills_jd = skills_jd.fillna('')
                skills_jd = skills_jd.to_numpy().tolist()
                skills_jd = [item for sublist in skills_jd for item in sublist]
            except:
                skills_jd = []
            try:
                skills_exp = pd.DataFrame(working_domains_and_skills_from_experience)
                skills_exp = skills_exp.fillna('')
                skills_exp = skills_exp.to_numpy().tolist()
                skills_exp = [item for sublist in skills_exp for item in sublist]
            except:
                skills_exp = []
            try:
                skills_resume = pd.DataFrame(working_domains_and_skills_from_resume)
                skills_resume = skills_resume.fillna('')
                skills_resume = skills_resume.to_numpy().tolist()
                skills_resume = [item for sublist in skills_resume for item in sublist]
            except:
                skills_resume = []
                
            # top 3 linkdin skills 
            linkedin_top_3 = linkedin_skill_list[:3]
            linkedin_top_3_dict = {}
            
            # similarity of top3 linkedin skills
            top3_similarity = 0
            for top in linkedin_top_3:
                skills_similarity_ = \
                        skills_similarity_with_jd(' '.join(skills_jd).lower().replace('skills',''), top.lower())
                linkedin_top_3_dict[top] = skills_similarity_
                top3_similarity += skills_similarity_
            print("top 3 similarity",top3_similarity)
            # linkedin skills without top 3
            if len(linkedin_skill_list)>3:
                linkedin_skills_from_ind_3 = \
                            skills_similarity_with_jd(' '.join(skills_jd).lower().replace('skills',''), ' '.join(linkedin_skill_list[3:]).lower())
                
            # skills from experience
            exp_skills_similarity_with_jd = \
            skills_similarity_with_jd(' '.join(skills_jd).lower().replace('skills',''), ' '.join(skills_exp).lower())
    
            # normalizing linkedin skills similarity
            total = top3_similarity+linkedin_skills_from_ind_3
            print("total",total)
            try:
                linkedin_skills_wo_top3 = linkedin_skills_from_ind_3/total
                top3_similarity = top3_similarity/total
            except:
                linkedin_skills_wo_top3 = 0
                top3_similarity = 0
                
            # 60% linkedin skills weightage
            linkedin_skills_weightage = (top3_similarity*(1/2)+exp_skills_similarity_with_jd*(1/3)+linkedin_skills_from_ind_3*(1/6))*0.6
            
            # skills from resume
            resume_skills_similarity_with_jd = \
            skills_similarity_with_jd(' '.join(skills_jd).lower().replace('skills',''), ' '.join(skills_resume).lower())
            
            # 40% resume skills weightage
            resume_skills_weightage = resume_skills_similarity_with_jd*0.4

            
            # final skills similarity
            skills_similarity_with_jd_ = resume_skills_weightage + linkedin_skills_weightage
            
            if not my_resume:
                linkedin_skills_weightage = (top3_similarity*(1/2)+exp_skills_similarity_with_jd*(1/3)+linkedin_skills_from_ind_3*(1/6))*1.0
                skills_similarity_with_jd_ = linkedin_skills_weightage
              
            if np.isnan(skills_similarity_with_jd_):
                print('aldksjfhaldjfa')
                skills_similarity_with_jd_ = 0
#             print('===================>>>>>>>>>.',skills_jd)
#             print('----------->>>>>>>',linkedin_skill_list)
#             skills_similarity_with_jd_ = \
#             skills_similarity_with_jd(' '.join(skills_jd).lower().replace('skills',''), ' '.join(linkedin_skill_list).lower())
            print(skills_similarity_with_jd_)
    
    
            profiles_info[profile] = {
                'Name': my_name,
                'skills from linkedin': linkedin_skill_list,
                'Experience Details': my_exp_details,
                'my_exp_description': my_exp,
                'domains_and_skills_from_linkedin': working_domains_and_skills_from_linkedin_skill_list,
                'working_domains_and_skills_from_experience': working_domains_and_skills_from_experience,
                'working_domains_and_skills_from_resume': working_domains_and_skills_from_resume,
                'working_domains_and_skills_from_jd': working_domains_and_skills_from_JD,
                'common_resume_jd_skills': common_resume_jd_skills,
                'common_linkedin_jd_skills': common_linkedin_jd_skills,
                'union_linkedin_resume_skills': union_linkedin_resume_skills,
                'skills_similarity_with_jd' : skills_similarity_with_jd_,
                }
    except Exception as e:

        print ('''This is exception in scoring_helpers.screening_main fun =====>>>> ''', e)
        return profiles_info
    return profiles_info