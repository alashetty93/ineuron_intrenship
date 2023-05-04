import pymongo
import pandas as pd
import json
import os
from dotenv import load_dotenv

load_dotenv()

client = pymongo.MongoClient(os.getenv("MONGO_DB_URL"))

database_name = "project"
collection_name = "shipment"


database = client[database_name]
collection = database[collection_name]

if __name__ == '__main__':

    # #import dataset to make dataframe
    # df = pd.read_csv('/config/workspace/SCMS_Delivery_History.csv')
    # print(df.shape)

    # #reset index of dataframe to have continuous count
    # df.reset_index(drop = True,inplace = True)

    # #convert df to json so that we can dump these records in MongoDB
    # json_records = list(json.loads(df.T.to_json()).values())

    # #inset json records into MongoDB
    # client[database_name][collection_name].insert_many(json_records)
