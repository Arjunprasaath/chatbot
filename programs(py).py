import tkinter
from tkinter import*
import nltk
import numpy as np
from nltk.stem import WordNetLemmatizer
lem =WordNetLemmatizer()
from keras.models import Sequential
import json
import random
from keras.models import load_model
import pickle
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
    

model=load_model('chatbot.h5')

top = tkinter.Tk()
top.title("Chatter")

intents = json.loads(open('intents.json').read())
words=pickle.load(open('word.pkl','rb'))
classes=pickle.load(open('classes.pkl','rb'))


message_text = tkinter.Text()
my_msg = tkinter.StringVar()
scrollbar = tkinter.Scrollbar(message_text) 
msg_list = tkinter.Listbox(message_text, height=20, width=50, yscrollcommand=scrollbar.set)
scrollbar.pack(side=tkinter.RIGHT, fill=tkinter.Y)
msg_list.pack(side=tkinter.LEFT, fill=tkinter.BOTH)
msg_list.pack()
message_text.pack()

def Enter_pressed(event):
    input_get = my_msg.get()
    input_get=input_get.lower()
    a=''
    while True:   
        if input_get!=a:
            if input_get == 'quit' or input_get=='exit':
                break
            else:
                for i in range(1):
                    msg_list.insert(i,'YOU:  %s ' % input_get)
                    my_msg.set('')
                    break
                q=con2num(input_get,words)
                result=model.predict(np.array(q))
                result_ind=np.argmax(result)
                print(result,result_ind)
                tag=classes[result_ind]
                print(tag)

                for t in intents['intents']:
                    if t['tag'] == tag:
                        response=t['responses']
                reply=str(random.choice(response))
                for i in range(1):
                    msg_list.insert(i,'BOT:  %s '% reply)
                return "break"

def con2num(sen,words):
    size=100
    encode=[one_hot(sen,size,filters='?!.:,()',lower=True)]
    pad_sen=pad_sequences(encode,maxlen=7,padding='post')
    print(pad_sen)
    return(pad_sen)

entry_field = tkinter.Entry(top, textvariable=my_msg)
entry_field.bind("<Return>", Enter_pressed)
entry_field.pack()
tkinter.mainloop()
