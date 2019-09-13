# 20180724 Customers Segmentation Engine / Arnaud ROUSSEAU

# This engine uses a .xls file with customers purchases descriptions to predict their 
# profiles just after the first shopping session.

# It uses a clean .csv file for the training.

#-----------------------------------------------------------------------------------------------------------------

# Libraries Import

import numpy as np
import pandas as pd

from sklearn import model_selection, preprocessing
from sklearn.ensemble import RandomForestClassifier

import datetime as dt

#-----------------------------------------------------------------------------------------------------------------

# Databases loading 

# Customer database for training 
data_client= pd.read_csv(
    'DATA_clean/data_Client.csv', 
    #sep='\t', 
    index_col=0,
    encoding='utf-8', 
    low_memory = False)


# Profiles to determine
data_client_pred = pd.read_excel('DATA_test/RFM_profiles.xlsx', sep=';')

# Test missing data
data_client_pred = data_client_pred.dropna(axis = 0 , subset=['Quantity']) 
data_client_pred = data_client_pred.dropna(axis = 0 , subset=['UnitPrice']) 
data_client_pred = data_client_pred.dropna(axis = 0 , subset=['CustomerID'])

if data_client_pred.empty:
    print("Some data in Excel file is missing !")
    
#-----------------------------------------------------------------------------------------------------------------

# New Columns: Total Price, Quantity clone
data_client['TotalPrice'] = data_client['Quantity'] * data_client['UnitPrice']
data_client_pred['TotalPrice'] = data_client_pred['Quantity'] * data_client_pred['UnitPrice']

#-----------------------------------------------------------------------------------------------------------------
# Customers Profiling with RFM Score for TRAINING

NOW = dt.datetime(2011,12,10)
data_client['InvoiceDate'] = pd.to_datetime(data_client['InvoiceDate'])

rfmTable = data_client.groupby('CustomerID').agg({'InvoiceDate': lambda x: (NOW - x.max()).days, 
                                          'InvoiceNo': lambda x: len(x.unique()), 
                                          'TotalPrice': lambda x: x.sum(), 
                                           'Quantity': lambda x: x.sum(),
                                           })

rfmTable['InvoiceDate'] = rfmTable['InvoiceDate'].astype(int)


rfmTable.rename(columns={'InvoiceDate': 'Recency', 
                         'InvoiceNo': 'Frequency', 
                         'TotalPrice': 'Monetary_value',
                         'Quantity': 'Total_Quant',
                        }, 
                         inplace=True)

#-----------------------------------------------------------------------------------------------------------------

# Parameters calculation of Customers for PREDICTION
rfmTable_pred = data_client_pred.groupby('CustomerID').agg({'TotalPrice': lambda x: x.sum(),
                                                            'Quantity': lambda x: x.sum()
                                                           })

rfmTable_pred.rename(columns={'TotalPrice': 'Monetary_value',
                              'Quantity': 'Total_Quant'
                             }, 
                         inplace=True)

#-----------------------------------------------------------------------------------------------------------------

# Recency score creation
def RScore(x,p,d):
    if x <= d[p][0.25]:
        return 1
    elif x <= d[p][0.50]:
        return 2
    elif x <= d[p][0.75]: 
        return 3
    else:
        return 4

#-----------------------------------------------------------------------------------------------------------------
    
# Frequency and Monetary Score creation   
def FMScore(x,p,d):
    if x <= d[p][0.25]:
        return 4
    elif x <= d[p][0.50]:
        return 3
    elif x <= d[p][0.75]: 
        return 2
    else:
        return 1

#-----------------------------------------------------------------------------------------------------------------

quantiles = rfmTable.quantile(q=[0.25,0.5,0.75])
quantiles = quantiles.to_dict()

rfmTable['r_quartile'] = rfmTable['Recency'].apply(RScore, args=('Recency',quantiles))
rfmTable['f_quartile'] = rfmTable['Frequency'].apply(FMScore, args=('Frequency',quantiles))
rfmTable['m_quartile'] = rfmTable['Monetary_value'].apply(FMScore, args=('Monetary_value',quantiles))

rfmTable['RFMScore'] = rfmTable.r_quartile.map(str) \
                            + rfmTable.f_quartile.map(str) \
                            + rfmTable.m_quartile.map(str)

rfmTable = rfmTable.filter(items=(['Recency',
                                   'Frequency',
                                   'Monetary_value',
                                   'Total_Quant',
                                   'RFMScore'
                                  ]))

rfm_unique = rfmTable['RFMScore'].unique()

#-----------------------------------------------------------------------------------------------------------------

# List of the 10 most popular profiles from prevous analysis
RFM_list_OK = ['111', '444', '443', '344', '222', '211', '122', '322', '244', '343']
rfmTable = rfmTable[rfmTable['RFMScore'].isin(RFM_list_OK)]

#-----------------------------------------------------------------------------------------------------------------

# Score RFM / Value association
def ScoreConv(x):
    if x == '111':
        return int(1)
    elif x == '444':
        return int(2)
    elif x == '443': 
        return int(3)
    elif x == '344': 
        return int(4)
    elif x == '222': 
        return int(5)
    elif x == '211': 
        return int(6)
    elif x == '122': 
        return int(7)
    elif x == '322': 
        return int(8)
    elif x == '244': 
        return int(9)
    elif x == '343': 
        return int(10)
    
#-----------------------------------------------------------------------------------------------------------------

rfmTable['RFM_Cat'] = rfmTable['RFMScore'].apply(lambda x: ScoreConv(x))
rfmTable['RFM_Cat'] = rfmTable['RFM_Cat'].astype(int)

#-----------------------------------------------------------------------------------------------------------------

# 1 = HardCore Buyer
# 2 = Middle Range Buyer
# 3 = Prospect

# Segment category / Value association
def ScoreConv2(x):
    if x == '111':
        return int(1)
    elif x == '444':
        return int(3)
    elif x == '443': 
        return int(3)
    elif x == '344': 
        return int(3)
    elif x == '222': 
        return int(2)
    elif x == '211': 
        return int(1)
    elif x == '122': 
        return int(2)
    elif x == '322': 
        return int(2)
    elif x == '244': 
        return int(3)
    elif x == '343': 
        return int(3)   
    
#-----------------------------------------------------------------------------------------------------------------

rfmTable['Profile_Cat'] = rfmTable['RFMScore'].apply(lambda x: ScoreConv2(x))
rfmTable['Profile_Cat'] = rfmTable['Profile_Cat'].astype(int)

#-----------------------------------------------------------------------------------------------------------------

# X train
data_client_rfm = rfmTable.filter(items=(['Total_Quant',
                                          'Monetary_value']))

scaler_client = preprocessing.StandardScaler().fit(data_client_rfm)
X_train = scaler_client.transform(data_client_rfm) 

# X for prediction
data_client_rfm_pred = rfmTable_pred.filter(items=(['Total_Quant',
                                                    'Monetary_value']))

scaler_client2 = preprocessing.StandardScaler().fit(data_client_rfm_pred)
X = scaler_client2.transform(data_client_rfm_pred)

# y train
y_train = rfmTable.filter(items=(['Profile_Cat']))

y_train = y_train['Profile_Cat'].ravel()

#-----------------------------------------------------------------------------------------------------------------

#Random Forest with best parameters
rfc = RandomForestClassifier(n_estimators= 100, max_depth = 3)
rfc.fit(X_train, y_train)

data_client_rfm_pred['Profile_Cat_Pred'] = rfc.predict(X)

#-----------------------------------------------------------------------------------------------------------------

# Score RFM / Value association
def ConvProfile(x):
    if x == 1:
        return str('Hardcore Buyer')
    elif x == 2:
        return str('Middle Range Buyer')
    elif x == 3: 
        return str('Prospect')
    
#-----------------------------------------------------------------------------------------------------------------

data_client_rfm_pred['Pred_Profile'] = data_client_rfm_pred['Profile_Cat_Pred'].apply(lambda x: ConvProfile(x))

#-----------------------------------------------------------------------------------------------------------------
# Results Display

print(data_client_rfm_pred)

#-----------------------------------------------------------------------------------------------------------------
# Results Saving

data_client_rfm_pred.to_csv(path_or_buf='DATA_test/data_Segmentation_Results.csv')