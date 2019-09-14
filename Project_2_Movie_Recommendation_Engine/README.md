# Movie Recommendation Engine

The aim of this project is to create a "Movie Recommendation Engine" for a internet site that starts with no client database.
The engine offers 5 movies using the "imdb-5000-movie-dataset".    
See https://www.imdb.com/

<b> - 20180529_Movie_Recommendations_Part_1_NoClean.ipynb	 </b>:   
This notebook is dedicated to the IMDb database cleaning and analysis.

<b> - 20180529_Movie_Recommendations_Part_2_Clean.ipynb	 </b>:   
This notebook is dedicated to the "Machine Learning" explorations with numerical variables (PCA, kMeans, DBSCAN, CAH).

<b> - 20180529_Movie_Recommendations_Part_3_Clean.ipynb	 </b>:   
Notebook dedicated to the ML explorations with alphanumerical variables.

<b> - 20180529_Movie_Recommendations_Part_Part_4_Mix.ipynb	 </b>:   
Notebook dedicated to the ML explorations with alphanumerical & numerical variables.

<b> - 20180529_Movie_Recom_Engine.py </b>:    
This MVP engine has been deployed using Flask.
Flask is a lightweight WSGI web application framework.
See https://palletsprojects.com/p/flask/

There is no web interface, the algo uses the URL.    
Add "/recommend/<int:ID_film>" at the end of your request, with "ID_film" the IMDB film number.    
It offers 5 movies from one IMDb reference.
