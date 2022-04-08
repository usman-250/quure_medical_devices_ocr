import os
import time
from bs4 import BeautifulSoup
import json


def parse_about_section(html):
    soup = BeautifulSoup(html)
    section = soup.find("section", {"class": "pv-about-section"})
    about = section.find("div",{"class":"inline-show-more-text"})
    about = about.getText().replace('\n','').replace('…','').replace('see more','').strip()
    return about
def parse_skills(html):
    skills = []
    soup = BeautifulSoup(html)
    section = soup.find("section",{"class":"pv-skill-categories-section"})
    skills_name = section.find_all("span",{"class":"pv-skill-category-entity__name-text"})
    for i in skills_name:
        skills.append(i.getText().replace('\n','').strip())
    return skills
def parse_experience(html):
    soup = BeautifulSoup(html)
    section = soup.find('section',{'id':'experience-section'})
    ul = section.find('ul',{'class':"pv-profile-section__section-info"})
    all_experiences = []
    for li in  ul.findAll('li',{'class':'pv-profile-section__list-item'}):
#         print(len(ul.findAll('li',{'class':'pv-profile-section__list-item'})))
        experience_dict ={}
        temp_list = []
        if not li.find('ul',{'class','pv-entity__position-group'}):
            if(li.find('h3')):
                experience_dict['Position'] = li.find('h3').getText().replace('Company Name','').replace('Title','').replace('\n','').strip()
            if li.find('p',{'class':'pv-entity__secondary-title'}):
                experience_dict['Company Name'] = li.find('p',{'class':'pv-entity__secondary-title'}).getText().replace('Company Name','').replace('\n','').strip()
            if li.find('h4',{'class','pv-entity__date-range'}):
                experience_dict['Employment'] = li.find('h4',{'class','pv-entity__date-range'}).getText().replace('Dates Employed','').replace('\n',' ').strip()
            if li.find('h4',{'class':'pv-entity__location'}):
                experience_dict['Location'] = li.find('h4',{'class':'pv-entity__location'}).getText().replace('Location','').replace('\n',' ').strip()
            if li.find('span',{'class','pv-entity__bullet-item-v2'}):
#                     all_h4 = li.find_all('h4')
#                     if len(all_h4)>2:
                experience_dict['Total Duration'] = li.find('span',{'class','pv-entity__bullet-item-v2'}).getText().replace('Employment Duration','').replace('\n','').strip()
            if li.find('div',{"class":"inline-show-more-text"}):
                experience_dict['Description'] = li.find('div',{"class":"inline-show-more-text"}).getText().replace('…','').replace('see more','').strip()
        if li.find('ul',{'class','pv-entity__position-group'}):
            sub_ul = li.find('ul',{'class','pv-entity__position-group'})
            if(li.find('h3')):
                experience_dict['Company Name'] = li.find('h3').getText().replace('Company Name','').replace('Title','').replace('\n','').strip()
            if(li.find('h3')):
                experience_dict['Total Duration'] = li.find('h4').getText().replace('Total Duration','').replace('Title','').replace('\n','').strip()
            for li in  sub_ul.findAll('li'):
                sub_categories = {}
                if(li.find('h3')):
                    sub_categories['Position'] = li.find('h3').getText().replace('Title','').replace('\n','').strip()
                if li.find('h4',{'class','pv-entity__date-range'}):
                    sub_categories['Employment'] = li.find('h4',{'class','pv-entity__date-range'}).getText().replace('Dates Employed','').replace('\n',' ').strip()
                if li.find('h4',{'class':'pv-entity__location'}):
                    sub_categories['Location'] = li.find('h4',{'class':'pv-entity__location'}).getText().replace('Location','').replace('\n',' ').strip()
                if li.find('span',{'class','pv-entity__bullet-item-v2'}):
#                     all_h4 = li.find_all('h4')
#                     if len(all_h4)>2:
                    sub_categories['Total Duration'] = li.find('span',{'class','pv-entity__bullet-item-v2'}).getText().replace('Employment Duration','').replace('\n','').strip()
                if li.find('div',{"class":"inline-show-more-text"}):
                    sub_categories['Description'] = li.find('div',{"class":"inline-show-more-text"}).getText().replace('…','').replace('see more','').strip()
                temp_list.append(sub_categories)
        if temp_list:
            experience_dict['sub_categories'] = temp_list
        all_experiences.append(experience_dict)
    return all_experiences


def parse_education(html):
    soup = BeautifulSoup(html)
    section = soup.find('section',{'id':'education-section'})
    ul = section.find('ul',{'class':"pv-profile-section__section-info"})
    all_education = []
    for li in  ul.findAll('li'):
        education_dict ={}
        if(li.find('h3')):
            education_dict['School Name'] = li.find('h3').getText().replace('\n','').strip()
        if li.find('p',{'class':'pv-entity__degree-name'}):
            education_dict['Degree Name'] = li.find('p',{'class':'pv-entity__degree-name'}).getText().replace('Degree Name','').replace('\n',' ').strip()
        if li.find('p',{'class':'pv-entity__fos'}):
            education_dict['Field of Study'] = li.find('p',{'class':'pv-entity__fos'}).getText().replace('Field Of Study','').replace('\n',' ').strip()
        if li.find('p',{'class':'pv-entity__dates'}):
            a = li.find_all('time')
            dates = []
            for i in a:
                dates.append(i.getText())
            dates = '-'.join(dates)
            education_dict['Date'] = dates
        all_education.append(education_dict)
    return all_education

def parse_contact_info(html):
    soup = BeautifulSoup(html)
    contact_info = {}
    if soup.find('section',{'class':'ci-vanity-url'}):
        section = soup.find('section',{'class':'ci-vanity-url'})
        if section.find('a',{'class':'pv-contact-info__contact-link'}):
            linkedin_link = section.find('a',{'class':'pv-contact-info__contact-link'})
            contact_info['linkedin_link'] = linkedin_link['href']
    if soup.find('section',{'class':'ci-phone'}):
        section = soup.find('section',{'class':'ci-phone'})
        if section.find('li',{'class':'pv-contact-info__ci-container'}):
            contact_info['mobile'] = section.find('li',{'class':'pv-contact-info__ci-container'}).getText().replace('\n','').replace('Mobile','').replace('()','').strip()
    if soup.find('section',{'class':'ci-email'}):
        section = soup.find('section',{'class':'ci-email'})
        if section.find('a',{'class':'pv-contact-info__contact-link'}):
            email = section.find('a',{'class':'pv-contact-info__contact-link'})
            contact_info['email'] = email['href'].replace('mailto:','')
    if soup.find('section',{'class':'ci-birthday'}):
        section = soup.find('section',{'class':'ci-birthday'})
        if section.find('span'):
            contact_info['birthday'] = section.find('span').getText().replace('\n','').strip()
    return contact_info

def parse_certifications(html):
    soup = BeautifulSoup(html)
    section = soup.find('section',{'id':'certifications-section'})
    ul = section.find('ul',{'class':'pv-profile-section__section-info'})
    all_certifications = []
    for li in  ul.findAll('li'):
        certification_dict = {}
        div = li.find('div',{'class':'pv-certifications__summary-info'})
        if div.find('h3'):
            title = div.find('h3').getText().replace('\n','').strip()
            certification_dict['title'] = title
        if div.find('p'):
            issued_by = div.find('p').getText().replace('\n','').replace('Issuing authority','').strip()
            certification_dict['issued_by'] = issued_by
        dates = div.find_all('p')
        date = dates[1].getText().split('\n')
        filtering_extra = filter(lambda x: x != "",date)
        filtered = list(filtering_extra)
        if 'Expires' in filtered[-1]:
            fil_date = filtered[-1].split('Expires')
            fil_date = '- Expires'.join(fil_date)
        else:
            fil_date = filtered[-1].split('No')
            fil_date = '- No'.join(fil_date)
        certification_dict['fil_date'] = fil_date
        all_certifications.append(certification_dict)
    return all_certifications

def parse_activity(html):
    soup  = BeautifulSoup(html)
    parent_div = soup.find('div',{'class':'pv-recent-activity-detail__outlet-container'})
    all_activity = []
    lenght_activity = len(parent_div.findAll('div',{'class':'occludable-update'}))
    if lenght_activity>50:
        lenght_activity = 50
        print(lenght_activity)
    
    for div in parent_div.findAll('div',{'class':'occludable-update'})[:lenght_activity]:
        temp_activity = {}
        person_name = ''
        actor_name = ''
        act_desc = ''
        time = ''
        text = ''
        if div.find('div',{'class':'feed-shared-actor'}):
            actor_div = div.find('div',{'class':'feed-shared-actor'})
        if div.find('div',{'class':'feed-shared-header__text-wrapper'}):
            person_name = div.find('div',{'class':'feed-shared-header__text-wrapper'}).getText().replace('\n','').strip()
        if actor_div.find('span',{'class':'feed-shared-actor__title'}):
            actor_name = actor_div.find('span',{'class':'feed-shared-actor__title'}).getText().replace('\n','').strip()
        if actor_div.find('span',{'class':'feed-shared-actor__description'}):
            act_desc = actor_div.find('span',{'class':'feed-shared-actor__description'}).getText().strip().replace('\n','')
        if actor_div.find('span',{'class':'feed-shared-actor__sub-description'}):
            time = actor_div.find('span',{'class':'feed-shared-actor__sub-description'}).getText().split('\n')[-2].strip()
        if div.find('div',{'class':'feed-shared-text'}):
            text = div.find('div',{'class':'feed-shared-text'}).getText().strip()
        if div.find('div',{'class':'comments-comments-list'}):
            comments_div = div.find('div',{'class':'comments-comments-list'})
            if comments_div.find('div',{'class':'feed-shared-text'}):
                comment = comments_div.find('div',{'class':'feed-shared-text'}).getText().strip().replace('\n','')
                temp_activity['comment'] = comment

        temp_activity['person_name'] = person_name
        temp_activity['actor_name'] = actor_name
        temp_activity['act_desc'] = act_desc
        temp_activity['time'] = time
        temp_activity['text'] = text
        all_activity.append(temp_activity)
    return all_activity



