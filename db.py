from time import sleep
import pymongo
from pymongo.collection import ReturnDocument


myclient = pymongo.MongoClient("mongodb://localhost:27017/")
mydb = myclient["qure"]

predictions_col = mydb["predictions"]
users_col = mydb["users"]
devices_col = mydb["devices"]

def insert_values(mycol,document):
    x = mycol.insert_many(document)
    return x.inserted_ids

def insert_doc(my_collection,doc):
    pred_inserted = my_collection.insert_one(doc)
    return pred_inserted


def return_all_data(collection):
    all_data = []
    for documents in collection.find():
        all_data.append(documents)
    return all_data

def return_all_users_email(collection):
    all_emails = []
    for documents in collection.find():
        all_emails.append(documents['user_email'])
    return all_emails

def find_documents_on_email(mycol,email):
    myquery = { "user_email": email }
    mydocs = mycol.find(myquery,{'_id':0})
    # return mydocs
    records = []
    for x in mydocs:
        records.append(x)
    if records:
        return records
    return {}

def update_device_doc(my_collection,key,data):
    # user_inserted = my_collection.update_one( { key[0]: data[key[0]], key[1]: data[key[1]], key[-1]: data[key[-1]]} , {'$set':data}, upsert=True)
    id = my_collection.find_one_and_update({ key[0]: data[key[0]], key[1]: data[key[1]], key[-1]: data[key[-1]]}, {
            '$setOnInsert': data
        }, upsert=True, return_document=ReturnDocument.AFTER)
    return id

def update_doc(my_collection,key,data):
    user_inserted = my_collection.find_one_and_update( { key: data[key]} , {'$setOnInsert':data},
                                             upsert=True, return_document=ReturnDocument.AFTER)
    return user_inserted

# def update_document(url,newvalues):
#     myquery = { 'url': url }
#     mycol.update_one(myquery, newvalues)
#     return mycol


# def update_document(url,newvalues):
#     myquery = { 'url': url }
#     x=find_document_on_url(mycol,url)
#     newvalues['$set']['scores'].update(x['scores'])
        
#     mycol.update_one(myquery, newvalues)
#     return mycol




    
