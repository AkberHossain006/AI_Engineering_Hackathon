import pandas as pd
import string
import re
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
string.punctuation
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import CountVectorizer

porter_stemmer = PorterStemmer()

from nltk.stem import WordNetLemmatizer
#defining the object for Lemmatization
wordnet_lemmatizer = WordNetLemmatizer()
def remove_punctuation(text):
    punctuationfree="".join([i for i in text if i not in string.punctuation])
    return punctuationfree
def tokenization(text):
  tokens = re.split(r'\s+', text)
  return tokens
def remove_stopwords(text):
    output= [i for i in text if i not in stopwords]
    return output

#defining a function for stemming
def stemming(text):
    stem_text = [porter_stemmer.stem(word) for word in text]
    return stem_text
def lemmatizer(text):
    lemm_text = [wordnet_lemmatizer.lemmatize(word) for word in text]
    return lemm_text

stopwords=nltk.corpus.stopwords.words('english')
df=pd.read_csv('data.csv',encoding="ISO-8859-1")
data=df[['description']]
data['clean_msg']= data['description'].apply(lambda x:remove_punctuation(x))
data['msg_lower']= data['clean_msg'].apply(lambda x:x.lower())
data['msg_tokenied']= data['msg_lower'].apply(lambda x: tokenization(x))
data['no_stopwords']= data['msg_tokenied'].apply(lambda x:remove_stopwords(x))
data['msg_stemmed']=data['no_stopwords'].apply(lambda x: stemming(x))

data['msg_lemmatized']=data['msg_stemmed'].apply(lambda x:lemmatizer(x))

data['processed'] = data['msg_lemmatized'].apply(lambda x: ' '.join(x))
print(data['processed'])
df['combined_text'] = df['name'] + ' ' + df['brand'] +' '+data['processed'] + ' ' + df['category']
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df['combined_text'])
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())
tfidf_df.to_csv("/content/sample_data/tfidf_output.csv", index=False)
!pip install faiss-cpu --no-cache-dir           
import faiss
import numpy as np
# Step 1: Convert DataFrame to numpy array
vectors = tfidf_df.to_numpy().astype('float32')  # FAISS requires float32

# Step 2: Create FAISS index (flat L2 index)
dimension = vectors.shape[1]
index = faiss.IndexFlatL2(dimension)

# Step 3: Add vectors to the index
index.add(vectors)

# Optional: Save the index to a file
faiss.write_index(index, "tfidf_faiss.index")

print("TF-IDF vectors stored in FAISS index!")
index = faiss.read_index("tfidf_faiss.index")

# Query: Find top 5 nearest neighbors of the first item
D, I = index.search(vectors[0:1], k=5)  # D = distances, I = indices
print("Top 5 similar items (by index):", I)
