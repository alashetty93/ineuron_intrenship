from shipment.pipeline.training_pipeline import  start_training_pipeline
from shipment.pipeline.batch_prediction import  start_batch_prediction
import os,sys
from shipment.exception import ShipmentException

file_path = "/config/workspace/SCMS_Delivery_History.csv"

if __name__ =='__main__':
    try:
        start_training_pipeline()
        # output_file = start_batch_prediction(input_file_path = file_path)
        # print(output_file)

    except Exception as e:
        ShipmentException(e,sys)