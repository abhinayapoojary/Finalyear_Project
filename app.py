# Import necessary libraries
import joblib
import re
import streamlit as st
import numpy as np
import pandas as pd
import pprint
import warnings
import tempfile
from io import StringIO
from PIL import  Image
from rake_nltk import Rake
import spacy
import spacy_streamlit
from collections import Counter
import en_core_web_sm
from nltk.tokenize import sent_tokenize
#from spacy.en import English
from spacy.lang.en import English

#nlp1 = spacy.load("en_core_web_sm")

# Warnings ignore 
warnings.filterwarnings(action='ignore')
st.set_option('deprecation.showfileUploaderEncoding', False)
st.set_option('deprecation.showPyplotGlobalUse', False)

# Import the custom modules 
import text_analysis as nlp
import text_summarize as ts

# Describing the Web Application 

# Title of the application 
st.title('Visualization and Summerization of Unstructured Text Data\n', )
st.subheader("by Abhinaya, Poorvika, Shravya, Akshata")

display = Image.open('images/source.jpg')
newsize =(750,500)
display = display.resize(newsize)
display = np.array(display)
st.image(display)
st.write("Source: Internet")

# Sidebar options
option = st.sidebar.selectbox('Navigation', 
["Home",
 "Word Cloud",
 "N-Gram Analysis",
 "Named Entity Recognition",
 "Text Summarizer"])

st.set_option('deprecation.showfileUploaderEncoding', False)

if option == 'Home':
	st.write(
        """
          ## Project Description
          Text Data is increasing rapidly In our day to day life. Getting what we want from this moulds 
          of unstructred data is becoming a problem. In cases like COVID the text are flooding in Internet. 
          Summerizing the data to get what we want and Visualizing the data is important for making any kind of analysis, 
          decision and so on. This tools will analyse this unstructured data and help in getting insights from this 
          data."""
          )

# Word Cloud Feature
elif option == "Word Cloud":

	st.header("Generate Word Cloud")
	st.subheader("Generate a word cloud from text containing the most popular words in the text.")

	# Ask for text or text file
	st.header('Enter text or upload file')
	text = st.text_area('Type Something', height=400) 
    
	#text_1 = st.file_uploader('Use Image Mask', type = ['txt'])
    
   
      
	# Upload mask image 
	mask = st.file_uploader('Use Image Mask', type = ['jpg']) 

    
	if st.button("Generate Wordcloud"):

		# Generate word cloud 
		st.write(len(text))
		nlp.create_wordcloud(text, mask)
		st.pyplot()

# N-Gram Analysis Option 
elif option == "N-Gram Analysis":
	
	st.header("N-Gram Analysis")
	st.subheader("This section displays the most commonly occuring N-Grams in your Data")

	# Ask for text or text file
	st.header('Enter text below')
	text = st.text_area('Type Something', height=400)

	# Parameters
	n = st.sidebar.slider("N for the N-gram", min_value=1, max_value=8, step=1, value=2)
	topk = st.sidebar.slider("Top k most common phrases", min_value=10, max_value=50, step=5, value=10)

	# Add a button 
	if st.button("Generate N-Gram Plot"): 
		# Plot the ngrams
		nlp.plot_ngrams(text, n=n, topk=topk)
		st.pyplot()
		

# Named Entity Recognition 
elif option == "Named Entity Recognition":
	st.header("Enter the statement that you want to analyze")

	st.markdown("**Random Sentence:** A Few Good Men is a 1992 American legal drama film set in Boston directed by Rob Reiner and starring Tom Cruise, Jack Nicholson, and Demi Moore. The film revolves around the court-martial of two U.S. Marines charged with the murder of a fellow Marine and the tribulations of their lawyers as they prepare a case to defend their clients.")
	text_input = st.text_area("Enter sentence")

    
	ner = en_core_web_sm.load()
	doc = ner(str(text_input))

	# Display 
	spacy_streamlit.visualize_ner(doc, labels=ner.get_pipe('ner').labels)



# Text Summarizer 
elif option == "Text Summarizer": 
	st.header("Text Summarization")
	
	st.subheader("Enter a corpus that you want to summarize")
	text_input = st.text_area("Enter a paragraph", height=150)
	sentence_count = len(sent_tokenize(text_input))
	st.write("Number of sentences:", sentence_count)
	
	model = st.sidebar.selectbox("Model Select", ["GenSim", "TextRank", "LexRank"])
	ratio = st.sidebar.slider("Select summary ratio", min_value=0.0, max_value=1.0, value=0.3, step=0.1)
	
	if st.button("Summarize"):
		if model == "GenSim":
			out = ts.text_sum_gensim(text_input, ratio=ratio)
			# st.write(out)
		elif model == "TextRank":
			out = ts.text_sum_text(text_input, ratio=ratio)
			# st.write(out)
		else:
			out = ts.text_sum_text(text_input, ratio=ratio)
			# st.write(out)

		st.write("**Summary Output:**", out)
		st.write("Number of output sentences:", len(sent_tokenize(out)))
