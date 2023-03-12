import sys
import pickle
from os import path,mkdir,system
import numpy as np
from sklearn.neural_network import MLPRegressor
import googlesearch
import wikipedia
import pyttsx3
import speech_recognition as sr
import tkinter as tk
import threading
import time

thresholdprobability = 0.9

root = tk.Tk()
root.title('Chatbot')

speech=pyttsx3.init()
speech.setProperty('voice',speech.getProperty("voices")[1].id)
speech.setProperty('rate',175)
rec=sr.Recognizer()
wikipedia.set_lang('en')
with open("intends.json") as f:
    jsond = eval(f.read())
words = []
doc_x = []
doc_y = []
tags = []
training = []
output = []

if path.exists("model\\intends.pickle"):
    f=open("model\\intends.pickle",'rb')
    words,tags,training,output=pickle.load(f)
    f.close()
else:
    for intend in jsond["intends"]:
        for pattern in intend["pattern"]:
            w=pattern.lower().replace('?','').split(' ')
            words.extend(w)
            doc_x.append(w)
            doc_y.append(intend["tag"])
        if intend["tag"] not in tags:
            tags.append(intend["tag"])
    words = sorted(list(set(words)))
    tags = sorted(tags)

    outempty = [0 for x in tags]

    for x,doc in enumerate(doc_x):
        wbag = []
        for w in words:
            if w in doc:
                wbag.append(1)
            else:
                wbag.append(0)
        doco = outempty[:]
        doco[tags.index(doc_y[x])]=1
        training.append(wbag)
        output.append(doco)

    training = np.array(training)
    output = np.array(output)
    mkdir("model")
    f=open("model\\intends.pickle",'wb')
    pickle.dump([words,tags,training,output],f)
    f.flush()
    f.close()


model = MLPRegressor(solver='lbfgs',alpha=1e-5,hidden_layer_sizes=(8,8),max_iter=1000)

if path.exists("model\\model.pickle"):
    f=open("model\\model.pickle",'rb')
    model = pickle.load(f)
    f.close()
else:
    model.fit(training[:],output[:])
    f = open("model\\model.pickle",'wb')
    pickle.dump(model,f)
    f.flush()
    f.close()

top = tk.Frame(root)
text = tk.Text(top)
text['state']='disabled'
text.pack(side='left',fill='both')
scrollbar = tk.Scrollbar(top,command=text.yview)
text['yscrollcommand']=scrollbar.set
scrollbar.pack(side='right',fill='y')
top.pack(side='top',fill='both')
bottom = tk.Frame(root)
entry = tk.Entry(bottom)
entry.grid(row=0,column=0,sticky='w')
hearb = tk.Button(bottom,text='\uD83C\uDF99',font=('Courier',16),bg='white',fg='blue')
hearb.grid(row=0,column=1,sticky='e')
enterb = tk.Button(bottom,text='\u27a5',font=('Courier',16),bg='white',fg='green')
enterb.grid(row=1,column=1,sticky='e')
bottom.pack()

def weighted_choice(lst,w):
    r = np.random.randint(1,sum(w)+1)
    for i,v in enumerate(w):
        r-=v
        if r<=0:
            return lst[i]

def hear():
        with sr.Microphone() as mic:
            audio = rec.listen(mic)
        return rec.recognize_google(audio,language='en-in')

def hear_insert():
    s = str(hear())
    entry.insert('end',s)
hearb['command']=hear_insert

speaking=False
def say(s):
    global speaking
    speaking=True
    speech.say(s)
    speech.runAndWait()
    speaking=False


def close():
    root.quit()
    root.destroy()
    quit(0)
    sys.exit()
root.protocol('WM_DELETE_WINDOW',close)

def bag_of_words(s):
    t = s.lower().replace('?','').split(' ')
    bag = [1 if w in t else 0 for w in words]
    return np.array(bag)

def predict_tag(s):
    p = model.predict([bag_of_words(s)])[0]
    t = "?"
    for i,v in enumerate(p):
        if v>thresholdprobability:
            t=tags[i]
            break
    return t

def get_output(t):
    outs = None
    weight = None
    if t=='?':
        outs = jsond["unknown"]["result"]
        weight = jsond["unknown"]["weight"]
    else:
        for i,v in enumerate(jsond["intends"]):
            if v["tag"] == t:
                outs = v["result"]
                weight = v["weight"]
                break
    out = weighted_choice(outs,weight)
    if out.find('<time>')!=-1:
        out = out.replace('<time>',time.strftime('%H:%M:%S'))
    if out.find('<date>')!=-1:
        out = out.replace('<date>',time.strftime('%a %d %b %Y'))
    return out


def enter_chat(param=None):
    if speaking:
        return
    text['state']='normal'
    inp = entry.get()
    entry.delete(0,'end')
    text.insert('end',"You: "+inp+"\n")
    inp = inp.lower()
    tag='?'
    out = 'Error'
    if inp.startswith('define'):
        d = inp[inp.index(' ')+1:]
        try:
            out = wikipedia.summary(d,auto_suggest=False,sentences=1)
        except Exception as e:
            out = str(e)
    elif inp.startswith('summarise') or inp.startswith('describe'):
        d = inp[inp.index(' ')+1:]
        try:
            out = wikipedia.summary(d,auto_suggest=False)
        except Exception as e:
            out = str(e)
    elif inp.startswith('search'):
        d = inp[inp.index(' ')+1:]
        try:
             s = list(googlesearch.search(d,num=2,stop=1,pause=2))[0]
             system('explorer "'+s+'"')
             out = "here's your result of web search:\n"+s
        except Exception as e:
            out = str(e)
    elif inp.startswith('run') or inp.startswith('open'):
        d = inp[inp.index(' ')+1:]
        thread = threading.Thread(target=system,args=(d,))
        thread.start()
        out = "running "+d
    else:
        tag = predict_tag(inp)
        out = get_output(tag)
    text.insert('end',"Chatbot: "+out+"\n")
    text['state']='disabled'
    thread = threading.Thread(target=say,args=(out,))
    thread.daemon=True
    thread.start()
    if tag=='bye':
        time.sleep(1)
        root.quit()
        root.destroy()
enterb['command']=enter_chat
entry.bind('<Return>',enter_chat)

root.mainloop()


