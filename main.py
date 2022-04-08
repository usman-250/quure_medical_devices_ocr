import sys,os 
# sys.path.append('./Flask_app/')

from flask import Flask, request, render_template, redirect,url_for,redirect,Blueprint
from flask_login import login_required, current_user, login_user
# from flask_session import Session

from .helper_functions import get_lcd, imgs_to_array
from werkzeug.utils import secure_filename
from keras.models import load_model
from .models import User
from flask import jsonify
from PIL import Image
import numpy as np
from .db import *
import datetime
import base64
import boto3
import time
import cv2
import io
import json



# records = return_all_data(mycol)
# print('\n\n---------->>>>>>> all records ', records,'\n\n')
mydict = {}


app=Flask(__name__)
main = Blueprint('main', __name__)

# main.config["SESSION_PERMANENT"] = False
# main.config["SESSION_TYPE"] = "filesystem"
# Session(main)

@main.route('/')
def index():
    # if not Session.get("name"):
    #     return redirect("/")
    return render_template('index.html')

@main.route('/upload')
def uploading():
    return render_template('uploading.html')

@main.route('/emails', methods=['GET', 'POST'])
@login_required
def show_emails():
    # print('\n\n\n\n=======================asdfadf', request.method)
    if request.method == 'POST':
        # print('---------->>>>>>>> in post')
        email = request.form['emailDropdown']
        # print(email)
        email_docs =  find_documents_on_email(predictions_col,email)
        all_emails = return_all_users_email(predictions_col)

        # print('=====docs ', email_docs)
        return render_template('show_emails.html', emails=list(set(all_emails)), data=email_docs)
    else:
        # print('---------->>>>>>>> in gwt')
        all_emails = return_all_users_email(mycol)
        return render_template('show_emails.html',emails=list(set(all_emails)))


# aws credentials
aws_textract = boto3.client(service_name='textract', region_name='us-east-2',aws_access_key_id = 'AKIAYO7JKT7XVYUKUWFN'
,aws_secret_access_key = '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')


#Load CNN model trained on data pre-defined in the paper
model=load_model('./Dataset/best_model.h5')

def predict_vals(files_add, path):
    # print('\n\n========',path)
    all_imgs_pred = {}
    for file in files_add:
        if file:
            filename = secure_filename(file.filename)
            file.save(filename)

            # Document
            documentName = filename
            global final_img_name
            final_img_name = filename
            #crop all regions

            if path == 'bp/td':
                preprocessed_img = get_lcd(documentName) # need some changes
                w, h=preprocessed_img.shape
                cv2.imwrite(filename + '_SP.jpg', preprocessed_img[0:int(h/2),0:w])
                cv2.imwrite(filename + '_DP.jpg', preprocessed_img[int(h/2):h,0:w])
                #convert img to ndarray and resize
                X_test = imgs_to_array( [filename+ '_SP.jpg',filename + '_DP.jpg'] )
                os.remove(filename+'_SP.jpg')
                os.remove(filename+'_DP.jpg')

            if path == 'glc/td':
                preprocessed_img = get_lcd(documentName) # need some changes
                w, h=preprocessed_img.shape
                # cv2.imwrite(filename + '_SP.jpg', preprocessed_img[0:int(h/2),0:w])
                cv2.imwrite(filename + '_DP.jpg', preprocessed_img[int(h/2):h,0:w])
                #convert img to ndarray and resize
                X_test = imgs_to_array( [filename+ '_DP.jpg'] )
                os.remove(filename+'_DP.jpg')

            if path == 'glc/md':
                preds = glucose_mobile(documentName)
                all_imgs_pred[documentName] = preds
                # print('\n\n --------------return')
                return all_imgs_pred, filename, True

            if path == 'temp/td':
                preprocessed_img = get_lcd(documentName) # need some changes
                w, h=preprocessed_img.shape
                # cv2.imwrite(filename + '_SP.jpg', preprocessed_img[0:int(h/2),0:w])
                cv2.imwrite(filename + '_DP.jpg', preprocessed_img[int(h/2):h,0:w])
                #convert img to ndarray and resize
                X_test = imgs_to_array( [filename+ '_DP.jpg'] )
                os.remove(filename+'_DP.jpg')
            # print('\n\n return')
            y_pred = model.predict( X_test )
            
            img_preds = []
            predicted_num = 0
            for i in range(X_test.shape[0]):
                pred_list_i = [np.argmax(pred[i]) for pred in y_pred]
                if path == 'glc/td':
                    predicted_num = str(pred_list_i[0])+str(pred_list_i[-1])
                elif path == 'temp/td':
                    predicted_num = str(pred_list_i[0])+str(pred_list_i[-1])
                else:
                    predicted_num = 100* pred_list_i[0] + 10 * pred_list_i[1] + 1* pred_list_i[2]
                    if predicted_num >= 1000:
                        predicted_num = predicted_num-1000

                img_preds.append(int(predicted_num))
                
            all_imgs_pred[documentName] = img_preds
            

    
    return all_imgs_pred, filename, False


def glucose_mobile(documentName):
    # Call Amazon Textract
    with open(documentName, "rb") as document:
        response = aws_textract.detect_document_text(
        Document={
            'Bytes': document.read(),
                }
            )
    # Print text

    # print('\n\n--->>>> response ',response)
    text = ""
    for item in response["Blocks"]:
        if item["BlockType"] == "LINE":
            # print ('\033[94m' +  item["Text"] + '\033[0m')
            text = text + " " + item["Text"]


    pos = text.find('mg/')
    # print('-------------->>>>>>>>>>>>\n',pos,text)
    text = text.replace('.',' ')
    final_text = [s for s in text[pos-10:pos].split() if s.isdigit()]
    return " ".join(final_text)[:4]



@main.route('/prediction', methods=['GET', 'POST'])
@login_required
def prediction():
    try:
        if request.method == 'POST':
            if request.form['deviceDropdown']:
                device_dropdown = request.form['deviceDropdown']

            if request.form['testDropdown']:
                test_dropdown = request.form['testDropdown']

            if request.files.getlist('myimage'):
                files_add = request.files.getlist("myimage")

            
            print('\n\n\n->>>>>>>>>>>>>>>',files_add, test_dropdown+'/'+device_dropdown )
            # return render_template('prediction.html')
            preds, filename, glc_mobile_device = predict_vals(files_add, test_dropdown+'/'+device_dropdown )

            print('\n\n=========>>>>>>>>>>\ Predictions : ',preds)

            # return render_template('loading.html')
        
        # mydict['email'] = current_user.email
        # mydict['time'] = str(datetime.datetime.now().time())
        # mydict['date'] = str(datetime.datetime.now().date())
        # mydict['test_name'] = test_dropdown
        # mydict['device_type'] = device_dropdown
        # mydict['image'] = {filename:''}


        ##################################################
        device_name = ""
        device_type = ""
        device_model = "c1"
        company_name = "Accu-Chek"
        user_name = current_user.name
        user_email = current_user.email
        predicted_at = str(datetime.datetime.now().time())
        updated_at = str(datetime.datetime.now().time())
        predicted_at_date = str(datetime.datetime.now().date())
        test_category = ''
        image = filename
        test_details = {}

        if device_dropdown == 'td':
            device_type = "table_device"
        elif device_dropdown == 'md':
            device_type = "mobile_device"
        
        if test_dropdown == 'glc':
            device_name = "glucco meter"
            test_category = "gluccos"
            test_details["gluccos"] = {"current_value": str(preds[filename][0]), "unit": "mg/dL"}
        elif test_dropdown == 'bp':
            device_name = "BP apparatus"
            test_category = "blood pressure"
            test_details["puls_rate"] = "72"
            test_details["upper"] = {"current_value": str(preds[filename][0]), "unit": "mmHg"}
            test_details["lower"] = {"current_value": str(preds[filename][-1]), "unit": "mmHg"}
        elif test_dropdown == 'temp':
            device_name = "thermometer"
            test_category = "tempreture"
            test_details["temp"] = {"current_value": str(preds[filename][0]), "unit": "°F"}

        
        test_details["time"] = "09:57:46"
        test_details["date"] = "2022-04-06"

        
        
        final_preds = make_final_dict(device_type, device_name,device_model,company_name,user_name,user_email,predicted_at_date,predicted_at,updated_at,\
            test_category,image,test_details)

        print('\n\n------->>>>>>>> : ', final_preds)


        ###################################################33
        

        # x = mycol.insert_one(mydict)

        im = Image.open(filename)
        data = io.BytesIO()
        rgb_im = im.convert('RGB')
        rgb_im.save(data, "JPEG")
        encoded_img_data = base64.b64encode(data.getvalue())
        os.remove(filename)
        # if glc_mobile_device:
        #     if preds.keys():
        #         preds = preds[list(preds.keys())[0]]
        #         # preds = ','.join([str(i) for i in preds[list(preds.keys())[0]]])
        # elif preds.keys():
        #         preds = ','.join([str(i) for i in preds[list(preds.keys())[0]]])
        return render_template('prediction.html',preds=json.dumps(final_preds), image=encoded_img_data.decode('utf-8') )
    except:
        return render_template('uploading.html',message = "unable to extract data")
        # return redirect(url_for('main.uploading'))


@main.route('/saving', methods=['GET', 'POST'])
@login_required
def saving():
    if request.method == 'POST':
        print("\n\n\n\n====>>>> fjldkjsd  request json",request.form['inppreds'])

        json_data = eval(request.form['inppreds'])

        print('\n\n type--->>> ',type(json_data))
        user_info = json_data.get('user')
        device_info = json_data.get('device')
        pred_info = json_data.get('prediction')
        

        #insert user info
        # user_inserted = users_col.insert_one(user_info, upsert=True)
        # users_col.update({ 'user_email': user_info['user_email']}), 

        user_inserted = users_col.update_one( { 'user_email': user_info['user_email']} , {'$set':user_info}, upsert=True)

     
        
        
        #insert device info
        # device_inserted = devices_col.insert_one(device_info, upsert=True)
        device_inserted = devices_col.update_one( { 'device_name': device_info['device_name']} , {'$set':device_info}, upsert=True)

        
        #insert prediction info
        pred_inserted = predictions_col.insert_one(pred_info.copy())

        # print("\n\n\n\n====>>>> request jason",request.json())
        # mydict['image'] = { final_img_name : request.form['preds'].split(',') }

        # x = mycol.insert_one(mydict.copy()) # check why  to use .copy error
        # print(x.inserted)

        return render_template('uploading.html')



def make_final_dict(device_type, device_name,device_model,company_name,user_name,user_email,predicted_at_date,predicted_at,updated_at,test_category,image,test_details):
    final_preds ={
        "device": {
            "device_name": device_name,
            "device_model": device_model,
            "company_name": company_name,
            "device_type": device_type
        },
        "user": {
            "user_name": user_name,
            "user_email": user_email,
            
        },
        "prediction": {
            "user_email": user_email,
            "device_model": device_model,
            "test_category": test_category,
            "image": image,
            "test_details":test_details,
            "time": {
                "predicted_at": predicted_at,
                "updated_at": updated_at,
                "predicted_at_date":predicted_at_date
                }
        }
    }
    return final_preds


if __name__ == '__main__':
    app.run(debug = True)



