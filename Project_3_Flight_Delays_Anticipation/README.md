# Flight Delays Anticipation    

<b> The average delay on all US flights exceeds 15 minutes (390 min).</b>     
The aim of this project is to anticipate a hypothetical flight delay, based on the BTS database (Bureau of Transportation Statistics, part of the United States Department of Transportation).    
See https://www.transtats.bts.gov/DataIndex.asp    

<b> - 20180704_Flight_Delays_Anticipation_P1NoClean.ipynb:</b>    
This notebook is dedicated to the BTS database cleaning and analysis.
 
<b> - 20180704_Flight_Delays_Anticipation_P2Clean_BaseX.ipynb: </b>    
This notebook is dedicated to the data engineering / Machine Learning phase.    
Creation of the pipeline after several explorations (Logistic regression, SVM, Ridge ...).    

<b> - flask_app.py </b>     
This MVP engine has been deployed using Flask.    
Flask is a lightweight WSGI web application framework.    
See https://palletsprojects.com/p/flask/

<b> - 20180710_README_API_CLOUD.txt: </b>    
How to use the MVP engine online (french version).    
There is no web interface, the algo uses the URL.
