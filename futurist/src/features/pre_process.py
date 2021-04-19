# Script to transform data for modelling

import pandas as pd
import numpy as np

def pre_process(data):
    output_column = "Selling_Price(lakhs)"
    unwanted_columns = ["name","engine","max_power","torque", "seats"]

    data['owner'] = data['owner'].map({'First Owner':1,'Second Owner':2,'Third Owner':3,"Fourth & Above":4})
    data = data.astype({"selling_price": "float64"})

    for index,row in data.iterrows():
        data.at[index,'selling_price'] /= 100000
        data.at[index,'present_price'] /= 100000
        data.at[index,'selling_price'] = np.round(data.at[index,'selling_price'],2)
        data.at[index,'present_price'] = np.round(data.at[index,'present_price'],2)

    data.rename(columns = {'selling_price':'Selling_Price(lakhs)','present_price':'Present_Price(lakhs)','owner':'Past_Owners', 'km_driven': 'Kms_Driven'},inplace = True)
    
    for _ in range(8):
        data = data[data['Present_Price(lakhs)'] < data['Present_Price(lakhs)'].quantile(0.99)]
        data = data[data['Selling_Price(lakhs)'] < data['Selling_Price(lakhs)'].quantile(0.99)]
        data = data[data['Kms_Driven'] < data['Kms_Driven'].quantile(0.99)]
    
    for val in ['seller_type:Trustmark Dealer','fuel:LPG']:
        query = val.split(":")
        data = data[data[query[0]] != query[1]]
    
    data['age'] = 2021 - data['year']
    data.drop('year',axis=1,inplace = True)
    data.drop_duplicates(subset=None, inplace=True)

    data.dropna(inplace=True)

    for column in unwanted_columns:
        data = data.drop(column,axis=1)
    
    
    for i, row in data.iterrows():
        data.at[i,'mileage'] = float(data.at[i,'mileage'].split(" ")[0])
    data = data.astype({"mileage": "float64"})
    
    data.to_csv("D:\Library\SEM VI\ML Lab\package\\futurist\data\processed\cleaned_dataset.csv", index=False)
    
    X = data
    Y = data[output_column]

    X = X.drop(output_column,axis=1)    
    X = pd.get_dummies(data = X,drop_first=True) 

    return X,Y


def pre_process_data(data):
    output_column = "Selling_Price(lakhs)"
    unwanted_columns = ["Car_Name","Selling_Price(lakhs)"]

    data.rename(columns = {'Selling_Price':'Selling_Price(lakhs)','Present_Price':'Present_Price(lakhs)','Owner':'Past_Owners'},inplace = True)
    for _ in range(3):
        data = data[data['Present_Price(lakhs)'] < data['Present_Price(lakhs)'].quantile(0.99)]
        data = data[data['Selling_Price(lakhs)'] < data['Selling_Price(lakhs)'].quantile(0.99)]
        data = data[data['Kms_Driven'] < data['Kms_Driven'].quantile(0.99)]
    data['age'] = 2021 - data['Year']
    data.drop('Year',axis=1,inplace = True)

    data.to_csv("D:\Library\SEM VI\ML Lab\package\\futurist\data\processed\cleaned_dataset.csv", index=False)

    X = data
    Y = data[output_column]

    for column in unwanted_columns:
        X = X.drop(column,axis=1)
    
    X = pd.get_dummies(data = X,drop_first=True) 

    return X,Y
