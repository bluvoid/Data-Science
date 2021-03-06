# Tags Proposal Engine

Tags auto generation, using the Stackexchange database.    
See https://data.stackexchange.com/stackoverflow/query/new    
Simple to use: Type a question related with programming (like on Stackoverflow) and it will generate 5 Tags.

In "Analyse" folder:    
<b> - 20180923_tag_proposal_Part1_NoClean.ipynb:</b>    
NLP pré processing with Beautifulsoup & NLTK (stopwords removal, Lemmatization);    

<b> - 20180923_tag_proposal_Part2_Clean.ipynb:</b>    
This notebook is dedicated to the data engineering / Machine Learning phase.    
Creation of the pipeline after several explorations (Frequency analysis, TF-IDF, Clustering).    

<b> - 20180925_tag_proposal_Part3_Clean_LDA_10000.ipynb:</b>   
This notebook is dedicated to the comparison between the previous pipeline & a LDA approch (Latent Dirichlet Allocation).

In "Site and App" folder:    
<b> - flask_app.py: </b>   
Final Tag proposal engine ("TFIDF + Clustering + Random Forest" or POS Tagging).    
This MVP engine has been deployed using Flask.    
Flask is a lightweight WSGI web application framework.    
See https://palletsprojects.com/p/flask/    
It uses a ".csv", generated by the previous notebooks, containing the clusters of each tags.    
  - In "templates" folder:    
  <b> - main_page.html </b>   
  Internet page of the engine.
