# -*- coding: utf-8 -*-

from google.colab import drive
drive.mount('/content/drive')

"""Importing Libraries"""

import os
import spacy
import math
import array
import pandas as pd
from tqdm import tqdm
import re

"""#Main Functions"""

def clean(s): #Function to clean the data
  clean_list = ['\n', '\n\n', '\n \n\n', '\n\n\n', "'", '“', '\n\n \n\n\n', '؟', '،', '!', ')', '(', '\n \n\n\n', ', ', '‘', '\n\n\n\n', '٪','۔','\n\n\n ','۔']
  s = [x for x in s if x not in clean_list]
  return s

def stopWordList(): #function to load and send stop word list
  stop_word_list = []
  stopword_file = open('/content/drive/MyDrive/NLP Online/Assignment#05/dataset/data.zip (Unzipped Files)/stopwords-ur.txt','r')
  stop_word_list = stopword_file.readlines()
  #removing \n form the words in the list
  stop_word_list =list(map(lambda s:s.strip(), stop_word_list ))
  return stop_word_list

def StopWordRemoval(stopWordsList_,list1):
   list_= [x for x in list1 if x not in stopWordsList_]
   return list_

def FilesCounter(path,category): #Function that will return total number of files in a directory

  TotalFiles= 0

  for base, dirs, files in os.walk(path):
      for Files in files:
          TotalFiles += 1
  print('Total number of files in ',category,': ',TotalFiles)
  return TotalFiles

def removeDuplicates(path): #Function to remove the duplicates for boolean naive classifier
  listOfData=[]
  for subdir, dirs, files in os.walk(path):
      for filename in files:
          filepath=""
          data__=""
          list_data=[]
          filepath = subdir + os.sep + filename
          f = open(filepath)
          data__ = f.read()
          data_token_ = unlp(data__)
          list_data = tokensInList(data_token_) #tokenizing testing data
          list_data = set(clean(list_data)) #removing duplicates from each review
          listOfData = listOfData + list(list_data) #concatenating list of each review
          
  return listOfData

def documentMaker(path): #Function to create single document of all files  
  data = ""
  countt=0
  for subdir, dirs, files in os.walk(path):
      for filename in files:
          filepath=""
          filepath = subdir + os.sep + filename
          file_ = open(filepath,'r')
          data = data+file_.read()
          data = data + " "
  return data

def tokensInList(text): #function to convert spacy tokens to list type
  list__ = []
  for token in text:
      list__.append(token.text)
  return list__

def frequency_dictionary(list__): #Function that returns a dictionary containing frequency of each words in the list
  frequency_distribution_dictionary = {}
  for item in list__:
    if (item in frequency_distribution_dictionary):
      frequency_distribution_dictionary[item] += 1
    else:
      frequency_distribution_dictionary[item] = 1
  return frequency_distribution_dictionary

def likelihoodGenerator(dict__,size_in_class,vocabulary_length): #Function that returns the dictionry contating the likelihood probabilities
                                                                 #of all words in the dictionary
  for key, value in dict__.items():                              
        dict__[key]= (value + 1) /(size_in_class + vocabulary_length) #calculating likelihood probabilites including laplace smoothing
  return dict__

"""classification function"""

def classifier(document_path,likelihood_dictionary_real,total_words_in_real,likelihood_dictionary_fake,total_words_in_fake,size_of_vocabulary,prior_prob_real,prior_prob_fake,duplication,stop_words,stop_wordList):
  laplace_smooth_real = 1/(total_words_in_real + size_of_vocabulary) # to handle zero case for real
  laplace_smooth_fake = 1/(total_words_in_fake + size_of_vocabulary) # to handle zero case for fake
  
  f = open(document_path)
  data__ = f.read()
  data_token_ = unlp(data__)
  list_data = tokensInList(data_token_) #tokenizing testing data
  list_data = clean(list_data)#cleaning testing data
  
  if (duplication == 'y'): #if it is true, remove duplicates first, for boolean naive bayes
    list_data = set(list_data)
    
  if (stop_words == 'y'): #if it is true, remove stop words first
    #removing
    list_data =StopWordRemoval(stop_wordList, list_data)

  count=0
  for i in list_data:
    count=count+1
    if count == 1:
      prob_prod= math.log( likelihood_dictionary_real.get(i,laplace_smooth_real)) #taking log to avoid underflow problem
    else:
      prob_prod= prob_prod+ math.log( likelihood_dictionary_real.get(i,laplace_smooth_real))#taking log to avoid underflow problem
      #print(" else: ", prob_prod)

  count = 0
  for i in list_data:
    count = count+1
    if count == 1:
      prob_prod2= math.log(likelihood_dictionary_fake.get(i,laplace_smooth_fake))#taking log to avoid underflow problem
    else: 
      prob_prod2= prob_prod2 + math.log(likelihood_dictionary_fake.get(i,laplace_smooth_fake))#taking log to avoid underflow problem




  prob_prod =prob_prod + math.log(prior_prob_real)#taking log to avoid underflow problem
  prob_prod2 = prob_prod2 + math.log( prior_prob_fake)#taking log to avoid underflow problem

  
  if (prob_prod>prob_prod2):
    return 1 #real
  else:
    return 0 #fake

"""Prediction Function"""

def predictedList(path,likelihood_dictionary_real,total_words_in_real,likelihood_dictionary_fake,total_words_in_fake,size_of_vocabulary,prior_prob_real,prior_prob_fake,duplication,stop_words,stop_wordList): #function that returns the list of predicted labels
  list__=[]
  for subdir, dirs, files in os.walk(path):
      for filename in files:
          filepath=""
          filepath = subdir + os.sep + filename
          predicted_label=classifier(filepath,likelihood_dictionary_real,total_words_in_real,likelihood_dictionary_fake,total_words_in_fake,size_of_vocabulary,prior_prob_real,prior_prob_fake,duplication,stop_words,stop_wordList )
          list__.append(predicted_label)
  return list__

"""Evaluation Matric Functions"""

def ConfusionMatrix_Generator(actual_labels_list_testset,real_test_predicted_labels,fake_test_predicted_labels): #Function to generate confusion Matrix
  groundtruth_and_prediction_dictionary = {'Actual_Labels': actual_labels_list_testset,
                                      'Predicted_labels': real_test_predicted_labels+fake_test_predicted_labels }
  df = pd.DataFrame(groundtruth_and_prediction_dictionary, columns=['Actual_Labels','Predicted_labels'])
  confusion_matrix = pd.crosstab(df['Actual_Labels'], df['Predicted_labels'], rownames=['Actual'], colnames=['Predicted'])
  return confusion_matrix

def All_in_one_matric_Generator(confusion_matrix): #Function that will take Confusion matrix as input and
                                                   #returns a dictionary that contains Accuracy,F1 Score, Precision
                                                   #Recall
  TN = confusion_matrix[0][0]
  TP = confusion_matrix[1][1]
  FN = confusion_matrix[0][1]
  FP = confusion_matrix[1][0]

  Accuracy = (TN+TP)/(TN+TP+FN+FP)
  Precision = TP/(TP+FP)
  Recall = TP/(TP+FN)
  F_one_score = (2*Precision*Recall)/(Precision+Recall)
  
  matric_dict = {'Accuracy': Accuracy,
                 'Precision': Precision,
                 'Recall': Recall,
                 'F1 Score': F_one_score}
  return matric_dict

"""# Loading And Pre-Processing the Training Data

Loading Training Set
"""

real_doc_string_train = documentMaker('/content/drive/MyDrive/NLP Online/Assignment#05/dataset/data.zip (Unzipped Files)/Train/Real')
fake_doc_string_train = documentMaker('/content/drive/MyDrive/NLP Online/Assignment#05/dataset/data.zip (Unzipped Files)/Train/Fake')

print(real_doc_string_train)

print(fake_doc_string_train)

"""Pre-Processing"""

#Tokenization
unlp = spacy.blank('ur')
realtext = unlp(real_doc_string_train)
faketext = unlp(fake_doc_string_train)

real_list_train= tokensInList(realtext)
fake_list_train= tokensInList(faketext)

#Data Cleaning
real_list_train = clean(real_list_train) #cleaing
fake_list_train = clean(fake_list_train) #cleaing

print(real_list_train)

print(fake_list_train)

"""# Loading And Pre-Processing the Testing Data

Loading Testing Set
"""

real_doc_string_test = documentMaker('/content/drive/MyDrive/NLP Online/Assignment#05/dataset/data.zip (Unzipped Files)/Test/Real')
fake_doc_string_test = documentMaker('/content/drive/MyDrive/NLP Online/Assignment#05/dataset/data.zip (Unzipped Files)/Test/Fake')

print(real_doc_string_test)

print(fake_doc_string_test)

"""Pre-Processing"""

#Tokenization
unlp = spacy.blank('ur')
realtext = unlp(real_doc_string_test)
faketext = unlp(fake_doc_string_test)

real_list_test= tokensInList(realtext)
fake_list_test= tokensInList(faketext)

#Data Cleaning
real_list_test = clean(real_list_test) #cleaing
fake_list_test = clean(fake_list_test) #cleaing

print(real_list_test)

print(fake_list_test)

"""#Prior Probabilities of Training data"""

number_of_real_files_train = FilesCounter('/content/drive/MyDrive/NLP Online/Assignment#05/dataset/data.zip (Unzipped Files)/Train/Real','real_train')
number_of_fake_files_train = FilesCounter('/content/drive/MyDrive/NLP Online/Assignment#05/dataset/data.zip (Unzipped Files)/Train/Fake','fake_train')

#Finding priors i-e P(real) and P(fake)
total_no_train = number_of_real_files_train + number_of_fake_files_train

prior_prob_real_train = number_of_real_files_train/total_no_train
prior_prob_fake_train = number_of_fake_files_train/total_no_train
print("Probability of class real in training set: ",prior_prob_real_train)
print("Probability of class fake in training set: ",prior_prob_fake_train)

"""# Actual Label List of Test Set"""

real_label_list=list(array.array('i',(1,)*FilesCounter('/content/drive/MyDrive/NLP Online/Assignment#05/dataset/data.zip (Unzipped Files)/Test/Real','test_real')))
fake_label_list = list(array.array('i',(0,)*FilesCounter('/content/drive/MyDrive/NLP Online/Assignment#05/dataset/data.zip (Unzipped Files)/Test/Fake','test_fake')))
actual_labels_list_testset = real_label_list + fake_label_list
print("1 represents Real")
print("2 represents Fake")
print(actual_labels_list_testset)

"""#Training and Testing Naive Bayes (With stopwords)

***Training Naive Bayes (With stopWords)***

Calculating Size of Vocabulary
"""

total_words_in_real = len(real_list_train)
total_words_in_fake = len(fake_list_train)
total_list = set(real_list_train + fake_list_train)# concatenating fake and passing to set function to make vocabulary of unique words
size_of_vocabulary = len(total_list)

"""Likelihood Probabilities """

real_frequency_dictionary = frequency_dictionary(real_list_train)
fake_frequency_dictionary = frequency_dictionary(fake_list_train)
#generating Likelihood Probabilities
likelihood_dictionary_real = likelihoodGenerator(real_frequency_dictionary,total_words_in_real,size_of_vocabulary)
likelihood_dictionary_fake = likelihoodGenerator(fake_frequency_dictionary,total_words_in_fake,size_of_vocabulary)

for key, value in likelihood_dictionary_real.items():
        print(key,value)

for key, value in likelihood_dictionary_fake.items():
        print(key,value)

"""***Testing Naive Bayes (With stopWords)***

Prediction
"""

#list containing the predicted labels (on test set's Real directory)
list___=[] #empty string, because we donot want to remove duplicates and stopwords
real_test_predicted_labels = predictedList('/content/drive/MyDrive/NLP Online/Assignment#05/dataset/data.zip (Unzipped Files)/Test/Real',
                                           likelihood_dictionary_real,total_words_in_real,likelihood_dictionary_fake,total_words_in_fake,
                                           size_of_vocabulary,prior_prob_real_train,prior_prob_fake_train,'n','n',list___)

#list containing the predicted labels (on test set's Fake directory)
fake_test_predicted_labels = predictedList('/content/drive/MyDrive/NLP Online/Assignment#05/dataset/data.zip (Unzipped Files)/Test/Fake',
                                           likelihood_dictionary_real,total_words_in_real,likelihood_dictionary_fake,total_words_in_fake,
                                           size_of_vocabulary,prior_prob_real_train,prior_prob_fake_train,'n','n',list___)

"""Evaluation"""

confusionMatrix = ConfusionMatrix_Generator(actual_labels_list_testset,real_test_predicted_labels,fake_test_predicted_labels)

confusionMatrix

"""Finding Accuracy, F1 score, Precision and Recall"""

matric_dict_naive_withStopwords = All_in_one_matric_Generator(confusionMatrix)
print(matric_dict_naive_withStopwords)

"""#Training and Testing Naive Bayes (Without stopwords)

***Training Naive Bayes (Without stopWords)***

Removing Stopwords
"""

#loading stopwords
stopWordsList_ = stopWordList()
#removing stopwords from the training set list
new_real_list_train=StopWordRemoval(stopWordsList_,real_list_train)
new_fake_list_train=StopWordRemoval(stopWordsList_,fake_list_train)

"""calculating size of vocabulary"""

total_words_in_real = len(new_real_list_train)
total_words_in_fake = len(new_fake_list_train)
total_list = set(new_real_list_train + new_fake_list_train)# concatenating fake and passing to set function to make vocabulary of unique words
size_of_vocabulary = len(total_list)

"""Likelihood Probabilities"""

real_frequency_dictionary = frequency_dictionary(new_real_list_train)
fake_frequency_dictionary = frequency_dictionary(new_fake_list_train)
#generating Likelihood Probabilities
likelihood_dictionary_real = likelihoodGenerator(real_frequency_dictionary,total_words_in_real,size_of_vocabulary)
likelihood_dictionary_fake = likelihoodGenerator(fake_frequency_dictionary,total_words_in_fake,size_of_vocabulary)

"""***Testing Naive Bayes (Without stopWords)***

loading stopword list
"""

#loading stopwords
stopWordsList_ = stopWordList()
#removing stopwords from the training set list
#new_real_list_test=StopWordRemoval(stopWordsList_,real_list_test)
#new_fake_list_test=StopWordRemoval(stopWordsList_,fake_list_test)

"""prediction"""

#list containing the predicted labels (on test set's Real directory)
list___=[] #empty string, because we donot want to remove duplicates and stopwords
real_test_predicted_labels = predictedList('/content/drive/MyDrive/NLP Online/Assignment#05/dataset/data.zip (Unzipped Files)/Test/Real',
                                           likelihood_dictionary_real,total_words_in_real,likelihood_dictionary_fake,total_words_in_fake,
                                           size_of_vocabulary,prior_prob_real_train,prior_prob_fake_train,'n','y',stopWordsList_)

#list containing the predicted labels (on test set's Fake directory)
fake_test_predicted_labels = predictedList('/content/drive/MyDrive/NLP Online/Assignment#05/dataset/data.zip (Unzipped Files)/Test/Fake',
                                           likelihood_dictionary_real,total_words_in_real,likelihood_dictionary_fake,total_words_in_fake,
                                           size_of_vocabulary,prior_prob_real_train,prior_prob_fake_train,'n','y',stopWordsList_)

"""Evaluation"""

confusionMatrix = ConfusionMatrix_Generator(actual_labels_list_testset,real_test_predicted_labels,fake_test_predicted_labels)

confusionMatrix

"""Finding Accuracy, F1 score, Precision and Recall"""

matric_dict_naive_withoutStopwords = All_in_one_matric_Generator(confusionMatrix)
print(matric_dict_naive_withStopwords)

"""#Training and Testing Boolean Naive Bayes (Without Duplicates)

***Training Boolean Naive Bayes***

Removing duplicates
"""

new_real_list_train=removeDuplicates('/content/drive/MyDrive/NLP Online/Assignment#05/dataset/data.zip (Unzipped Files)/Train/Real')
new_fake_list_train=removeDuplicates('/content/drive/MyDrive/NLP Online/Assignment#05/dataset/data.zip (Unzipped Files)/Train/Fake')

"""Calculating size of vocabulary"""

total_words_in_real = len(new_real_list_train)
total_words_in_fake = len(new_fake_list_train)
total_list = set(new_real_list_train + new_fake_list_train)# concatenating fake and passing to set function to make vocabulary of unique words
size_of_vocabulary = len(total_list)

"""likelihood probablities"""

real_frequency_dictionary = frequency_dictionary(new_real_list_train)
fake_frequency_dictionary = frequency_dictionary(new_fake_list_train)
#generating Likelihood Probabilities
likelihood_dictionary_real = likelihoodGenerator(real_frequency_dictionary,total_words_in_real,size_of_vocabulary)
likelihood_dictionary_fake = likelihoodGenerator(fake_frequency_dictionary,total_words_in_fake,size_of_vocabulary)

"""***Testing Boolean Naive Bayes (Without Duplicates)***

Prediction
"""

#list containing the predicted labels (on test set's Real directory)
list___=[] #empty string, because we donot want to remove duplicates and stopwords
real_test_predicted_labels = predictedList('/content/drive/MyDrive/NLP Online/Assignment#05/dataset/data.zip (Unzipped Files)/Test/Real',
                                           likelihood_dictionary_real,total_words_in_real,likelihood_dictionary_fake,total_words_in_fake,
                                           size_of_vocabulary,prior_prob_real_train,prior_prob_fake_train,'y','n',list___)

#list containing the predicted labels (on test set's Fake directory)
fake_test_predicted_labels = predictedList('/content/drive/MyDrive/NLP Online/Assignment#05/dataset/data.zip (Unzipped Files)/Test/Fake',
                                           likelihood_dictionary_real,total_words_in_real,likelihood_dictionary_fake,total_words_in_fake,
                                           size_of_vocabulary,prior_prob_real_train,prior_prob_fake_train,'y','n',list___)

"""Evaluation"""

confusionMatrix = ConfusionMatrix_Generator(actual_labels_list_testset,real_test_predicted_labels,fake_test_predicted_labels)

confusionMatrix

"""Finding Accuracy, F1 score, Precision and Recall"""

matric_dict_boolean_naive_bayes = All_in_one_matric_Generator(confusionMatrix)
print(matric_dict_naive_withStopwords)

"""#Comparison of Naive Bayes (with/without stopwords) and Boolean Naive Bayes"""

matric_dict_naive_withStopwords

matric_dict_naive_withoutStopwords

matric_dict_boolean_naive_bayes

"""By seeing the results, we can say that the boolean naive bayes classifier out performs. Because its F1 score, precision, and Recall is closer to 1 and also we can see that, these measures are also greater with the measures of other models that are naive bayes with stopwords and naive bayes without stopwords.

# Visualizing the results
"""

import numpy as np
import matplotlib.pyplot as plt
data = [[matric_dict_naive_withStopwords['Accuracy']*100, matric_dict_naive_withStopwords['Precision']*100, matric_dict_naive_withStopwords['Recall']*100,matric_dict_naive_withStopwords['F1 Score']*100],
[matric_dict_naive_withoutStopwords['Accuracy']*100, matric_dict_naive_withoutStopwords['Precision']*100, matric_dict_naive_withoutStopwords['Recall']*100,matric_dict_naive_withoutStopwords['F1 Score']*100],
[matric_dict_boolean_naive_bayes['Accuracy']*100, matric_dict_boolean_naive_bayes['Precision']*100, matric_dict_boolean_naive_bayes['Recall']*100,matric_dict_boolean_naive_bayes['F1 Score']*100]]
X = np.arange(4)
fig = plt.figure(figsize=(8,5))
ax = fig.add_axes([0,0,1,1])
plt.bar(X + 0.00, data[0], color = 'b', width = 0.25,label ='naive bayes with stopwords')
plt.bar(X + 0.25, data[1], color = 'g', width = 0.25,label = 'naive bayes without stopwords')
plt.bar(X + 0.50, data[2], color = 'r', width = 0.25,label = 'boolean naive bayes')

plt.xticks([r + barWidth for r in range(len(data[0]))],
        ['Accuracy', 'Precision', 'Recall', 'F1 Score'])
plt.legend()
plt.show()

"""Also we can see in the bar plot, that boolean naive bayes out performs. Because its accuracy is highest, its precision is approximately equivalent to the precision of other models. The F1 score of boolean naive bayes is also highest.

# Extra credit work

what i am going to do is, i will concatenate the negation word and the word that is next to it. Like if a sentence is, "mujhay in ka pizza pasand nahi aya". so instead of calculating probability of "nahi" and "aya" alone, i will calculate the probability of "nahi aya" together.
"""

def negationConcat(list_,neg_words):
  ctr = 0
  new_word=""
  for word in list_:
    if (ctr+1<len(list_)):
      if list_[ctr+1] in neg_words: #if neg word found, concatinate it with the next word to it
        new_word=word+" "+list_[ctr+1]
        list_.pop(ctr)#remove word at index ctr
        list_.pop(ctr)#remove next word, i-e at index ctr now
        list_.insert(ctr,new_word)#inset the concatinated word
        new_word = ""
    ctr+=1
  return list_

list_of_negation_words = ['نہیں','نہ','نا'] #we can icrease this list

"""concatenating the negation word with the next word to it"""

n_real_list_train = negationConcat(real_list_train,list_of_negation_words)
n_fake_list_train = negationConcat(fake_list_train,list_of_negation_words)

"""Calculating size of vocabulary"""

total_words_in_real = len(n_real_list_train)
total_words_in_fake = len(n_fake_list_train)
total_list = set(n_real_list_train + n_fake_list_train)# concatenating fake and passing to set function to make vocabulary of unique words
size_of_vocabulary = len(total_list)

"""***Training the model***

likelihood probabilities
"""

real_frequency_dictionary = frequency_dictionary(n_real_list_train)
fake_frequency_dictionary = frequency_dictionary(n_fake_list_train)
#generating Likelihood Probabilities
likelihood_dictionary_real = likelihoodGenerator(real_frequency_dictionary,total_words_in_real,size_of_vocabulary)
likelihood_dictionary_fake = likelihoodGenerator(fake_frequency_dictionary,total_words_in_fake,size_of_vocabulary)

"""***Testing the model***

classifier and prediction

some parameters of function and code needs changes to implement this extra work. so i have written new functions
"""

def classifier_extra(document_path,likelihood_dictionary_real,total_words_in_real,likelihood_dictionary_fake,total_words_in_fake,size_of_vocabulary,prior_prob_real,prior_prob_fake,neg_words_):
  laplace_smooth_real = 1/(total_words_in_real + size_of_vocabulary) # to handle zero case for real
  laplace_smooth_fake = 1/(total_words_in_fake + size_of_vocabulary) # to handle zero case for fake
  
  f = open(document_path)
  data__ = f.read()
  data_token_ = unlp(data__)
  list_data = tokensInList(data_token_) #tokenizing testing data
  list_data = clean(list_data)#cleaning testing data
  list_data = negationConcat (list_data,neg_words_)

  count=0
  for i in list_data:
    count=count+1
    if count == 1:
      prob_prod= math.log( likelihood_dictionary_real.get(i,laplace_smooth_real)) #taking log to avoid underflow problem
    else:
      prob_prod= prob_prod+ math.log( likelihood_dictionary_real.get(i,laplace_smooth_real))#taking log to avoid underflow problem
      #print(" else: ", prob_prod)

  count = 0
  for i in list_data:
    count = count+1
    if count == 1:
      prob_prod2= math.log(likelihood_dictionary_fake.get(i,laplace_smooth_fake))#taking log to avoid underflow problem
    else: 
      prob_prod2= prob_prod2 + math.log(likelihood_dictionary_fake.get(i,laplace_smooth_fake))#taking log to avoid underflow problem




  prob_prod =prob_prod + math.log(prior_prob_real)#taking log to avoid underflow problem
  prob_prod2 = prob_prod2 + math.log( prior_prob_fake)#taking log to avoid underflow problem

  
  if (prob_prod>prob_prod2):
    return 1 #real
  else:
    return 0 #fake

def predictedList_extra(path,likelihood_dictionary_real,total_words_in_real,likelihood_dictionary_fake,total_words_in_fake,size_of_vocabulary,prior_prob_real,prior_prob_fake,list_of_negation_words): #function that returns the list of predicted labels
  list__=[]
  for subdir, dirs, files in os.walk(path):
      for filename in files:
          filepath=""
          filepath = subdir + os.sep + filename
          predicted_label=classifier_extra(filepath,likelihood_dictionary_real,total_words_in_real,likelihood_dictionary_fake,total_words_in_fake,size_of_vocabulary,prior_prob_real,prior_prob_fake,list_of_negation_words )
          list__.append(predicted_label)
  return list__

"""Prediction"""

#list containing the predicted labels (on test set's Real directory)
list___=[] #empty string, because we donot want to remove duplicates and stopwords
real_test_predicted_labels = predictedList_extra('/content/drive/MyDrive/NLP Online/Assignment#05/dataset/data.zip (Unzipped Files)/Test/Real',
                                           likelihood_dictionary_real,total_words_in_real,likelihood_dictionary_fake,total_words_in_fake,
                                           size_of_vocabulary,prior_prob_real_train,prior_prob_fake_train,list_of_negation_words)

#list containing the predicted labels (on test set's Fake directory)
fake_test_predicted_labels = predictedList_extra('/content/drive/MyDrive/NLP Online/Assignment#05/dataset/data.zip (Unzipped Files)/Test/Fake',
                                           likelihood_dictionary_real,total_words_in_real,likelihood_dictionary_fake,total_words_in_fake,
                                           size_of_vocabulary,prior_prob_real_train,prior_prob_fake_train,list_of_negation_words)

"""Evaluation"""

confusionMatrix = ConfusionMatrix_Generator(actual_labels_list_testset,real_test_predicted_labels,fake_test_predicted_labels)

confusionMatrix

matric_dict_naive_ExtraCredit = All_in_one_matric_Generator(confusionMatrix)
print(matric_dict_naive_withStopwords)

"""Previous best was boolean naive bayes, so compairing this extra work with boolean naive bayes"""

matric_dict_boolean_naive_bayes

"""so, more or less, both models(boolean naive and naive extra credit) are good. But the precision,F1 score and recall of boolean naive bayes is slighter higher"""