from textblob import TextBlob

text = TextBlob("I am in a very good mood right now, so happy.")
#print(text)

#Polarity
#print(text.sentiment[0])



#Text Summarization

from gensim.summarization.summarizer import summarize 
from gensim.summarization import keywords 


#Need more than one sentence
randomText = "In an attempt to build an AI-ready workforce, Microsoft announced Intelligent Cloud Hub which has been launched to empower the next generation of students with AI-ready skills. Envisioned as a three-year collaborative program, Intelligent Cloud Hub will support around 100 institutions with AI infrastructure, course content and curriculum, developer support, development tools and give students access to cloud and AI services. As part of the program, the Redmond giant which wants to expand its reach and is planning to build a strong developer ecosystem in India with the program will set up the core AI infrastructure and IoT Hub for the selected campuses. The company will provide AI development tools and Azure AI services such as Microsoft Cognitive Services, Bot Services and Azure Machine Learning.According to Manish Prakash, Country General Manager-PS, Health and Education, Microsoft India, said With AI being the defining technology of our time, it is transforming lives and industry and the jobs of tomorrow will require a different skillset. This will require more collaborations and training and working with AI. That’s why it has become more critical than ever for educational institutions to integrate new cloud and AI technologies. The program is an attempt to ramp up the institutional set-up and build capabilities among the educators to educate the workforce of tomorrow. The program aims to build up the cognitive skills and in-depth understanding of developing intelligent cloud connected solutions for applications across industry. Earlier in April this year, the company announced Microsoft Professional Program In AI as a learning track open to the public. The program was developed to provide job ready skills to programmers who wanted to hone their skills in AI and data science with a series of online courses which featured hands-on labs and expert instructors as well. This program also included developer-focused AI school that provided a bunch of assets to help build AI skills."
summWords = summarize(randomText)
#print(summWords)



# NER
import spacy
from spacy import displacy
from collections import Counter
import en_core_web_sm
nlp = en_core_web_sm.load()

doc = nlp(randomText)
entities = []
entityLabels = []
for ent in doc.ents:
    entities.append(ent.text)
    entityLabels.append(ent.label_)

res = dict(zip(entities, entityLabels)) 
#print(res)
#keys_for_ORG = [k for k in res if res[k] == "ORG"]
#print(keys_for_ORG)


def entRecognizer(entDict, typeEnt):
    entList = [k for k in entDict if entDict[k] == typeEnt]
    return entList

print(entRecognizer(res,"ORG"))

#print(entities)
#print(entityLabels)
#print(len(entities))
#print(len(entityLabels))

from nltk.tokenize import sent_tokenize
sents = sent_tokenize(randomText)
#print(len(sents))