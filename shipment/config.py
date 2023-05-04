import pymongo
import json
import os
from dataclasses import dataclass
  
@dataclass
class EnvironmentVariable:

    #provide the mongodb url to connect python to mongodb.
    mongodb_url:str = os.getenv("MONGO_DB_URL")

#create an instance of Environment class
env_variable = EnvironmentVariable()

#create instance of MongoClient for connection
mongo_client = pymongo.MongoClient(env_variable.mongodb_url)

TARGET_COLUMN = "Freight Cost (USD)"