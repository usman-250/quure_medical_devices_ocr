import sys,os 
# sys.path.append('./Flask_app/')

from flask import Flask, request, render_template, redirect,url_for,redirect,Blueprint
from flask_login import login_required, current_user, login_user
# from flask_session import Session

from .helper_functions import get_lcd, imgs_to_array
from werkzeug.utils import secure_filename
from keras.models import load_model
from bson.objectid import ObjectId
from .models import User
from flask import jsonify
from PIL import Image
import numpy as np
from .db import *
import datetime
import base64
import boto3
import time
import json
import cv2
import io




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
    if request.method == 'POST':
        email = request.form['emailDropdown']
        email_docs = find_documents_on_email(predictions_col,email)
        all_emails = return_all_users_email(predictions_col)

        return render_template('show_emails.html', emails=list(set(all_emails)), data=email_docs)
    else:
        all_emails = return_all_users_email(predictions_col)
        return render_template('show_emails.html',emails=list(set(all_emails)))


# aws credentials
aws_textract = boto3.client(service_name='textract', region_name='us-east-2',aws_access_key_id = 'AKIAYO7JKT7XVYUKUWFN'
,aws_secret_access_key = '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')


#Load CNN model trained on data pre-defined in the paper
model=load_model('./Dataset/best_model.h5')

def predict_vals(files_add, path):
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

    text = ""
    for item in response["Blocks"]:
        if item["BlockType"] == "LINE":
            text = text + " " + item["Text"]


    pos = text.find('mg/')
    text = text.replace('.',' ')
    final_text = [s for s in text[pos-10:pos].split() if s.isdigit()]
    print('\n\n\n\n---------->>>>>>><<<<<<<<<<>>>>>>> ',final_text)
    # return " ".join(final_text)[:3]
    return final_text


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

        ##################################################
        device_name = ""
        device_type = ""
        device_model = ""
        device_make = ""
        device_launch_date = ""
        device_varrient = ""
        company_name = "Accu-Chek"

        user_name = current_user.name
        user_email = current_user.email
        user_role = ""

        predicted_at = str(datetime.datetime.utcnow()) # utc time
        updated_at = str(datetime.datetime.utcnow())
        # predicted_at_date = str(datetime.datetime.now().date())
        test_category = ''
        image_path = filename
        test_details = {}

        if device_dropdown == 'td':
            device_type = "table_device"
        elif device_dropdown == 'md':
            device_type = "mobile_device"
        
        if test_dropdown == 'glc':
            device_name = "glucco meter"
            test_category = "gluccos"
            test_details[test_category] = {}
            test_details[test_category]["gluccos"] = {"current_value": str(preds[filename][0]), "unit": "mg/dL"}
            test_details["time"] = "09:57:46"
            test_details["date"] = "2022-04-06"
        elif test_dropdown == 'bp':
            device_name = "BP apparatus"
            test_category = "blood pressure"
            test_details["puls_rate"] = {"current_value": "72", "unit": ""}
            test_details[test_category] = {}
            test_details[test_category]["upper"]= {"current_value": str(preds[filename][0]), "unit": "mmHg"}
            test_details[test_category]["lower"] = {"current_value": str(preds[filename][-1]), "unit": "mmHg"}
            test_details["time"] = "09:57:46"
            test_details["date"] = "2022-04-06"
        elif test_dropdown == 'temp':
            device_name = "thermometer"
            test_category = "tempreture"
            test_details[test_category] = {}
            test_details[test_category]["temp"] = {"current_value": str(preds[filename][0]), "unit": "°F"}

        
        final_preds = make_final_dict(device_make,device_launch_date,device_varrient, device_type, device_name,device_model,company_name,user_role,user_name,user_email,predicted_at,updated_at,\
            test_category,image_path,test_details)

        #################################
        user_info = final_preds.get('user')
        #insert user info
        user_id = update_doc(users_col,'user_email',user_info)


        device_info = final_preds.get('device')
        #insert device info
        device_id = update_device_doc(devices_col,['device_name','device_type','device_model'],device_info)

        print('\n\n\n===>>>>>>>>>>> ', str(user_id['_id']))
        final_preds["prediction"]["user_id"] = str(user_id['_id'])
        final_preds["prediction"]["device_id"] = str(device_id['_id'])
        #############################################
        


        ###################################################33


        im = Image.open(filename)
        data = io.BytesIO()
        rgb_im = im.convert('RGB')
        rgb_im.save(data, "JPEG")
        encoded_img_data = base64.b64encode(data.getvalue())
        os.remove(filename)
        return render_template('prediction.html',preds=json.dumps(final_preds), image=encoded_img_data.decode('utf-8') )
    except Exception as e:
        os.remove(final_img_name)
        print("\n\n-------->>> ",e)
        return render_template('uploading.html',message = "unable to extract data, Try with another image")


@main.route('/saving', methods=['GET', 'POST'])
@login_required
def saving():
    if request.method == 'POST':
        # print("\n\n====>>>> fjldkjsd  request json",request.form['inppreds'])

        json_data = eval(request.form['inppreds'])

        # print('\n\n type--->>> ', type(json_data))
        # user_info = json_data.get('user')
        # device_info = json_data.get('device')
        pred_info = json_data.get('prediction')
        pred_info['user_id'] = ObjectId(pred_info['user_id'])
        pred_info['device_id'] = ObjectId(pred_info['device_id'])

        # #insert user info
        # user_inserted = update_doc(users_col,'user_email',user_info)
        # # user_inserted = users_col.update_one( { 'user_email': user_info['user_email']} , {'$set':user_info}, upsert=True)
        
        # #insert device info
        # device_inserted = update_device_doc(devices_col,['device_name','device_type','device_model'],device_info)
        # # device_inserted = devices_col.update_one( { 'device_name': device_info['device_name']} , {'$set':device_info}, upsert=True)

        #insert prediction info
        pred_inserted = insert_doc(predictions_col,pred_info)
        # pred_inserted = predictions_col.insert_one(pred_info.copy())


        return render_template('uploading.html')



def make_final_dict(device_make,device_launch_date,device_varrient, device_type, device_name,device_model,company_name, user_role, user_name,user_email,predicted_at,updated_at,test_category,image_path,test_details):
    
    if device_name == "glucco meter":
        device_model ='G1'
    elif device_name == "BP apparatus":
        device_model ='B1'
    elif device_name == "thermometer":
        device_model ='T1'

    final_preds ={
        "device": {
            "device_name": device_name,
            "device_model": device_model,
            "company_name": company_name,
            "device_type": device_type,
            "device_make": device_make,
            "device_launch_date" : device_launch_date,
            "device_varrient" : device_varrient
        },
        "user": {
            "user_name": user_name,
            "user_email": user_email,
            "user_role" : user_role
            
        },
        "prediction": {
            "user_id": "",
            "device_id": "",
            "test_category": test_category,
            "image_path": image_path,
            "test_details":test_details,
            "time": {
                "predicted_at": predicted_at,
                "updated_at": updated_at,
                # "predicted_at_date":predicted_at_date
                }
        }
    }
    return final_preds


if __name__ == '__main__':
    app.run(debug = False)



