#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
import numpy as np
import nltk
nltk.download('wordnet')
import re
from bs4 import BeautifulSoup
import warnings
warnings.filterwarnings('ignore')


# In[8]:


get_ipython().system(" pip install bs4 # in case you don't have it installed")

# Dataset: https://s3.amazonaws.com/amazon-reviews-pds/tsv/amazon_reviews_us_Beauty_v1_00.tsv.gz


# ## Read Data

# In[1]:


import pandas as pd
df = pd.read_table('amazon_reviews_us_Beauty_v1_00.tsv', on_bad_lines='skip')
df["review_body"] = df['review_body'].astype(str) + df["review_headline"].astype(str)


# In[ ]:





# ## Keep Reviews and Ratings

# In[2]:


t = df[["star_rating","review_body"]]


#  ## We form three classes and select 20000 reviews randomly from each class.
# 
# 

# In[99]:


conditions = [
    (t['star_rating'] == '1') | (t['star_rating'] == '2'),
    (t['star_rating'] == '3'),
    (t['star_rating'] == '4') | (t['star_rating'] == '5')
]
choices = ['class_1', 'class_2','class_3']
t['classes'] = np.select(conditions,choices, default=0)


t1 = t[t.classes == "class_1"].sample(n=20000)
t2 = t[t.classes == "class_2"].sample(n=20000)
t3 = t[t.classes == "class_3"].sample(n=20000)


# In[100]:


t4 = pd.DataFrame(columns = ['star_rating', 'review_body', 'classes'])
t4 = t4.append(t1, ignore_index = True)
t4 = t4.append(t2, ignore_index = True)
t4 = t4.append(t3, ignore_index = True)


# # Data Cleaning
# 
# 

# In[102]:


result4 = t4['review_body'].str.len().mean()

t4['review_body'] = t4['review_body'].str.lower()
import re
t4['review_body'] = t4['review_body'].apply(lambda x: re.split('https:\/\/.*', str(x))[0])
t4['review_body'] = t4['review_body'].str.replace(r'<[^<>]*>', '', regex=True)
t4['review_body'].str.strip()

import contractions
t4["review_body"] = t4['review_body'].apply(lambda x: [contractions.fix(word) for word in x.split()])
t4['review_body'] = [','.join(map(str, l)) for l in t4['review_body']]
t4 = t4.replace(',',' ', regex=True)


t4['review_body'] = t4['review_body'].str.replace('[^a-zA-Z]', ' ', regex=True).str.strip()

min_threshold_rep = 1
t4['review_body']= t4['review_body'].str.replace(r'(\w)\1{%d,}'%(min_threshold_rep-1), r'\1')
t4["review_body"] = t4["review_body"].str.replace(r"\s+(.)\1+\b", "").str.strip()
t4['review_body'] = t4['review_body'].str.replace(r'\b(\w+)(\s+\1)+\b', r'\1')


result5 = t4['review_body'].str.len().mean()

print("Average length before and after cleaning" ,result4,",",result5)


# # Pre-processing

# ## remove the stop words 

# In[103]:


result6 = t4["review_body"].apply(len).mean()


from nltk.corpus import stopwords
nltk.download('stopwords')

stop = stopwords.words('english')
t4['review_body'] = t4['review_body'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))


# ## perform lemmatization  

# In[104]:


import nltk
nltk.download('omw-1.4')
from nltk.stem import WordNetLemmatizer
lmtzr = WordNetLemmatizer()
t4['review_body'] = t4['review_body'].apply(lambda word: lmtzr.lemmatize(word, pos='v'))

result7 = t4["review_body"].apply(len).mean()
print("Average length before and after cleaning" ,result6,",",result7)


# # TF-IDF Feature Extraction

# In[92]:



from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report

tfidf = TfidfVectorizer(max_features = None, ngram_range = (1,5), analyzer = 'char')

X1 = tfidf.fit_transform(t4['review_body'])
y1 = t4['classes']

corpus = []
for x in t4["review_body"]:
    corpus.append(x)

cv = TfidfVectorizer(ngram_range = (1,1))
x = cv.fit_transform(corpus)
#print(cv.vocabulary_)


# In[93]:


all_feature_names = cv.get_feature_names_out()
for word in all_feature_names:
    indx = cv.vocabulary_.get(word)
    #print(f"{word} {cv.idf_[indx]}")


# In[94]:


X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size = 0.2, random_state=0) 


# # Perceptron

# In[95]:


from sklearn.datasets import load_digits
from sklearn.linear_model import Perceptron
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score   
from sklearn.metrics import recall_score   


clf = Perceptron()
clf.fit(X1_train, y1_train)

print("\n\n  Perceptron model:")


y1_pred = clf.predict(X1_test)
print("          [class1              class2              class3]               average")
print("precision", precision_score(y1_test, y1_pred,average = None).tolist(), "," , "%.2f" %  precision_score(y1_test, y1_pred, average = 'macro'))
print("f1       ", f1_score(y1_test, y1_pred,average = None).tolist(), "," , "%.2f" %  f1_score(y1_test, y1_pred, average = 'macro'))
print("recall   " , recall_score(y1_test, y1_pred,average = None).tolist(), "," , "%.2f" %  recall_score(y1_test, y1_pred, average = 'macro'))

print("\n \n")
#print(f1_score(y1_test, y1_pred))
#print 'Recall:', recall_score(y_test, prediction)
#print 'Precision:', precision_score(y_test, prediction)

print(classification_report(y1_test, y1_pred))


# # SVM

# In[96]:


clf = LinearSVC()

clf.fit(X1_train, y1_train)
y1_pred = clf.predict(X1_test)

print("\n Logistic Regression:")

print("          [class1              class2              class3]               average")
print("precision", precision_score(y1_test, y1_pred,average = None).tolist(), "," , "%.2f" %  precision_score(y1_test, y1_pred, average = 'macro'))
print("f1       ", f1_score(y1_test, y1_pred,average = None).tolist(), "," , "%.2f" %  f1_score(y1_test, y1_pred, average = 'macro'))
print("recall   " , recall_score(y1_test, y1_pred,average = None).tolist(), "," , "%.2f" %  recall_score(y1_test, y1_pred, average = 'macro'))

print("\n \n")
print(classification_report(y1_test, y1_pred))


# # Logistic Regression

# In[97]:


from sklearn.linear_model import LogisticRegression 

logreg =  LogisticRegression() 

logreg.fit(X1_train,y1_train) 
y1_pred = clf.predict(X1_test)
print("\n Logistic Regression:")
print("          [class1              class2              class3]               average")
print("precision", precision_score(y1_test, y1_pred,average = None).tolist(), "," , "%.2f" %  precision_score(y1_test, y1_pred, average = 'macro'))
print("f1       ", f1_score(y1_test, y1_pred,average = None).tolist(), "," , "%.2f" %  f1_score(y1_test, y1_pred, average = 'macro'))
print("recall   " , recall_score(y1_test, y1_pred,average = None).tolist(), "," , "%.2f" %  recall_score(y1_test, y1_pred, average = 'macro'))

print("\n \n")
print(classification_report(y1_test, y1_pred))


# # Naive Bayes

# In[98]:


from sklearn.datasets import load_iris
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LinearRegression

gnb = MultinomialNB()

gnb.fit(X1_train, y1_train)

y1_pred = gnb.predict(X1_test)

print("\n Naive Bayes:")

print("          [class1              class2              class3]               average")
print("precision", precision_score(y1_test, y1_pred,average = None).tolist(), "," , "%.2f" %  precision_score(y1_test, y1_pred, average = 'macro'))
print("f1       ", f1_score(y1_test, y1_pred,average = None).tolist(), "," , "%.2f" %  f1_score(y1_test, y1_pred, average = 'macro'))
print("recall   " , recall_score(y1_test, y1_pred,average = None).tolist(), "," , "%.2f" %  recall_score(y1_test, y1_pred, average = 'macro'))

print("\n \n")
print(classification_report(y1_test, y1_pred))


# In[ ]:




