import json
import nltk
import numpy as np
from nltk.stem import WordNetLemmatizer
lem=WordNetLemmatizer()
import tensorflow
import keras
from keras.models import Sequential
from keras.layers import Dense,Embedding,LSTM,Flatten
import pickle



'''extracting the data set'''
with open('intents.json','r') as f:
    data=json.load(f)

'''assigning the data'''
words=[]
document=[]
classes=[]
ign=['?','.','!',',']
for intent in data['intents']:
    for pattern in intent['patterns']:
        w=nltk.word_tokenize(pattern)
        words.extend(w)
        document.append((w,intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])
doc=[]
for d in document:
	doc.append(d[0])


'''szorting the data'''
words=[lem.lemmatize(w.lower())for w in words if w not in ign]
words=sorted(list(set(words)))
classes=sorted(list(set(classes)))

lists=[]
def reemovNestings(l): 
    for i in l: 
        if type(i[0]) == list: 
            reemovNestings(i) 
        else:
            a=''
            for j in i:
                a=a+' '+ j
            lists.append(a)
reemovNestings(doc)


'''creating files'''

pickle.dump(words,open('word.pkl','wb'))
pickle.dump(classes,open('classes.pkl','wb'))

'''training'''

training=[]
output=[0]*len(classes)
def reemovNestings(l): 
    for i in l: 
        if type(i[0]) == list: 
            reemovNestings(i) 
        else:
            a=''
            for j in i:
                a=a+' '+ j
            output.append(a)
reemovNestings(doc)
print(output)
max_len=7
encoded_words=[]
encoded_words=[one_hot(w, size,filters='!?',lower=True) for w in lists]
padded_sentence=pad_sequences(encoded_words,max_len,padding='post')

for doc in document:
    out=list(output)
    out[classes.index(doc[1])]=1
    training.append([out])


training=np.array(training)
train_y=list(training)
train_y=np.array(train_y)


'''model creation'''

model=Sequential()
model.add(Embedding(input_dim=100,output_dim=200,input_length=max_len))
model.add(LSTM(128,return_sequences=True))
model.add(Dense(128,activation='relu'))
model.add(LSTM(128,return_sequences=True))
model.add(Dense(128,activation='relu'))
model.add(LSTM(128,return_sequences=True))
model.add(Dense(128,activation='relu'))
model.add(Flatten())
model.add(Dense(len(classes),activation='softmax'))

model.compile(loss='categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])

model.fit(padded_sentence,train_y,epochs=100,verbose=1)

model.save('chatbot.h5')
    


