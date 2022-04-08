import sys,os 
# sys.path.append('./Flask_app/')

from flask import Flask, request, render_template, redirect,url_for,redirect,Blueprint
from flask_login import login_required, current_user, login_user
# from flask_session import Session

from .helper_functions import get_lcd, imgs_to_array
from werkzeug.utils import secure_filename
from keras.models import load_model
from .models import User
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



records = return_all_data(mycol)
print('\n\n---------->>>>>>> all records ', records,'\n\n')
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
    print('\n\n\n\n=======================asdfadf', request.method)
    if request.method == 'POST':
        print('---------->>>>>>>> in post')
        email = request.form['emailDropdown']
        # print(email)
        email_docs =  find_documents_on_email(mycol,email)
        all_emails = return_all_users_email(mycol)

        print('=====docs ', email_docs)
        return render_template('show_emails.html', emails=list(set(all_emails)), data=email_docs)
    else:
        print('---------->>>>>>>> in gwt')
        all_emails = return_all_users_email(mycol)
        return render_template('show_emails.html',emails=list(set(all_emails)))


# aws credentials
aws_textract = boto3.client(service_name='textract', region_name='us-east-2',aws_access_key_id = 'AKIAYO7JKT7XVYUKUWFN'
,aws_secret_access_key = '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')


#Load CNN model trained on data pre-defined in the paper
model=load_model('./Dataset/best_model.h5')

def predict_vals(files_add, path):
    print('\n\n========',path)
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
                print('\n\n --------------return')
                return all_imgs_pred, filename, True

            if path == 'temp/td':
                preprocessed_img = get_lcd(documentName) # need some changes
                w, h=preprocessed_img.shape
                # cv2.imwrite(filename + '_SP.jpg', preprocessed_img[0:int(h/2),0:w])
                cv2.imwrite(filename + '_DP.jpg', preprocessed_img[int(h/2):h,0:w])
                #convert img to ndarray and resize
                X_test = imgs_to_array( [filename+ '_DP.jpg'] )
                os.remove(filename+'_DP.jpg')
            print('\n\n return')
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

            print('\n\n=========>>>>>>>>>>\nFinal Predictions : ',preds)

            # return render_template('loading.html')
        
        mydict['email'] = current_user.email
        mydict['time'] = str(datetime.datetime.now().time())
        mydict['date'] = str(datetime.datetime.now().date())
        mydict['test_name'] = test_dropdown
        mydict['device_type'] = device_dropdown
        mydict['image'] = {filename:''}
        

        # x = mycol.insert_one(mydict)

        im = Image.open(filename)
        data = io.BytesIO()
        rgb_im = im.convert('RGB')
        rgb_im.save(data, "JPEG")
        encoded_img_data = base64.b64encode(data.getvalue())
        os.remove(filename)
        if glc_mobile_device:
            if preds.keys():
                preds = preds[list(preds.keys())[0]]
                # preds = ','.join([str(i) for i in preds[list(preds.keys())[0]]])
        elif preds.keys():
                preds = ','.join([str(i) for i in preds[list(preds.keys())[0]]])
        return render_template('prediction.html',preds=preds, image=encoded_img_data.decode('utf-8') )
    except:
        return render_template('uploading.html',message = "unable to extract data")
        # return redirect(url_for('main.uploading'))


@main.route('/saving', methods=['GET', 'POST'])
@login_required
def saving():
    if request.method == 'POST':
        # print("request forms",request.form['preds'])
        mydict['image'] = { final_img_name : request.form['preds'].split(',') }

        x = mycol.insert_one(mydict.copy()) # check why  to use .copy error
        # print(x.inserted)

    return render_template('uploading.html' )



def make_final_dict(device_name,device_model,company_name,user_name,user_email,predicted_at,updated_at,\
    test_category,image,test_details):
    {
        "device": {
            "device_name": device_name,
            "device_model": device_model,
            "company_name": company_name
        },
        "user": {
            "user_name": user_name,
            "user_email": user_email,
            "time": {
            "predicted_at": predicted_at,
            "updated_at": updated_at
            }
        },
        "prediction": {
            "test_category": test_category,
            "image": image,
            "test_details": {
            "upper": {
                "current_value": test_details.get("upper").get("current_value"),
                "unit": test_details.get("upper").get("unit"),
            },
            "lower": {
                "current_value": test_details.get("lower").get("current_value"),
                "unit": test_details.get("lower").get("unit"),
            },
            "puls_rate": test_details.get("puls_rate"),
            "time": test_details.get("time"),
            "date": test_details.get("date")
            }
        }
    }


if __name__ == '__main__':
    app.run(debug = True)