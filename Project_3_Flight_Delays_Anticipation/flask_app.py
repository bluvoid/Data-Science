# Moteur d'anticipation de retard de vol avec Regression Logistique

from flask import Flask, request
import numpy as np
import pandas as pd

from collections import Counter
import itertools

from sklearn import preprocessing, decomposition, cluster, metrics
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import linear_model

from itertools import cycle


# Flask
app = Flask(__name__)


@app.route('/')
def rien():
    rien = "Init Flask OK, Projet4: vous devez taper une commande de la forme anti/MONTH/DAY_OF_MONTH/DAY_OF_WEEK/CRS_DEP_TIME/CRS_ARR_TIME/AIR_TIME/FLIGHTS/DISTANCE"
    return str(rien)


@app.route('/anti/<int:MONTH>/<int:DAY_OF_MONTH>/<int:DAY_OF_WEEK>/<int:CRS_DEP_TIME>/<int:CRS_ARR_TIME>/<int:AIR_TIME>/<int:FLIGHTS>/<int:DISTANCE>')     
def moteur(MONTH, DAY_OF_MONTH, DAY_OF_WEEK, CRS_DEP_TIME, CRS_ARR_TIME, AIR_TIME, FLIGHTS, DISTANCE):
    vol = [MONTH, DAY_OF_MONTH, DAY_OF_WEEK, CRS_DEP_TIME, CRS_ARR_TIME, AIR_TIME, FLIGHTS, DISTANCE]
    pred = lrXG_CV.predict([vol])
    
    if pred[0] == 0:
        resultat="Ce vol sera à l'heure avec jusqu'à 14 minutes de plus par rapport a l'heure indiquée."
    
    if pred[0] == 1:
        resultat='Ce vol sera en retard de 15 minutes minimum.'
        
    return str(resultat) 


# fonction Moteur Recommandation
def antic_carrier(ID_CARRIER):
    result = lrXG_CV.predict_proba(data_plane_CLEAN_X.loc[ID_CARRIER])
    shape = result.shape
    proba_ponct = (sum(result[:,0])) / shape[0]
    proba_ponct = proba_ponct*100

    return proba_ponct 


@app.route('/anticipation/<string:ID_CARRIER>')
def test(ID_CARRIER):
    if ID_CARRIER not in data_plane_CLEAN_X.index:
        resultat="ERREUR: cet ID_CARRIER n'est pas dans la base de donnée ! Les ID sont 'NK','OO','DL','AS','WN','B6','HA','AA','UA','EV','F9','VX'."
    else:  
        proba_ponct = antic_carrier(ID_CARRIER)
        proba_ponct = format(proba_ponct, '.2f')
        resultat = "Le CARRIER " + ID_CARRIER + " à une probabilité de "+ proba_ponct +"% d'être à l'heure (avec jusqu'à 14 minutes de plus par rapport a l'heure indiquée)."
      
    return str(resultat)


# ouverture de la base BTS déjà concaténée et samplée
data_plane_CLEAN_X= pd.read_csv(
    '/home/ArnaudRousseau/mysite/DATA_cleanX/data_plane_CLEAN_X_full.csv',
    index_col=0,
    encoding='utf-8',
    low_memory = False)


# ouverture de la base y7
data_plane_CLEAN_y_7_full= pd.read_csv(
    '/home/ArnaudRousseau/mysite/DATA_cleanX/data_plane_CLEAN_y_7_full.csv',
    index_col=0,
    encoding='utf-8',
    low_memory = False)


# preprocessing
scalerX = preprocessing.StandardScaler().fit(data_plane_CLEAN_X)
X_scaled = scalerX.transform(data_plane_CLEAN_X) 

# split
#X_train, X_test, y_train, y_test = train_test_split(X_scaled,
#                                                   data_plane_CLEAN_y_7_full['DEP_DEL15'], 
#                                                   test_size=0.25)           
           
lrXG_CV = linear_model.LogisticRegressionCV()

# On entraîne ce modèle sur les données d'entrainement
lrXG_CV.fit(data_plane_CLEAN_X, data_plane_CLEAN_y_7_full)