import uvicorn
from fastapi import FastAPI, File, UploadFile
from house_prices import HousePrice
import numpy as np
import joblib
import pandas as pd
from regression_model.processing.features import factorizeTransformer
from typing import List
import csv
import codecs
from regression_model.processing.features import get_data
# from regression_model.predict import make_prediction
# from features import heatEncTransformer
# from features import get_data

app = FastAPI()

pickle_in = open('grid.pkl','rb')

grid_model = joblib.load(pickle_in)
feat_list = ['bed','bathroom','year_built','heating','Property_type','area','county','zipcode']

@app.get('/')
def index():
    return {"welcome to the house price prediction page!!"}


@app.post("/uploadfiles/")
# async def create_upload_files(files: List[UploadFile] = File(...)):
async def create_upload_files(csv_file: UploadFile = File(...)):
    # csv_reader = csv.reader(codecs.iterdecode(csv_file.file,'utf-8'))
    csv_reader = pd.read_csv(csv_file.file)
    df = get_data(csv_reader)
    df = df[feat_list]
    predictions = grid_model.predict(df)
    results = {
     "predictions": [np.exp(pred) for pred in predictions]
    }
    return results
    print(df)
    # return {"filenames": [file.filename for file in files]}


@app.post('/predict')
def predict_house(data:HousePrice):
    data = data.dict()
    bed=  data['bed']
    bathroom = data['bathroom']
    year_built=  data['year_built']
    heating = data['heating']
    Property_type = data['Property_type']
    area = data['area']
    county = data['county']
    zipcode = data['zipcode']


    mydata = pd.DataFrame([data])
    # df = create_upload_files()
    # print(df)
    predictions = grid_model.predict([[bed,bathroom,year_built,heating,Property_type,area,county,zipcode]])
    results = {
      'predictions': [np.exp(pred) for pred in predictions]
    }

    return results


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
