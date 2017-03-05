'''
THis script is to find out the various bigrams and trigrams that are present in the given WO.The idea is to find out if
by looking at the various words we can figure out what could be the key issue or the item.
If we apply the knowldege from association like which words have a close relation with conditions than the presence of
that asset and condition might tell us that they are the main thigs.
'''
from nltk.tokenize import WordPunctTokenizer
from nltk.collocations import BigramCollocationFinder
from nltk.collocations import TrigramCollocationFinder
from nltk.metrics import TrigramAssocMeasures
from nltk.metrics import BigramAssocMeasures
import numpy as np
from nltk.util import bigrams
from nltk.util import trigrams
import csv
from nltk.corpus import stopwords
import re
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()

import pandas as pd


prefixpath="/Users/305015992/pythonProjects/assetAnswer/"


arrPumpWords=[]

assetWordsFile=prefixpath+"pumpKeywords1.csv"
for dictLine in open(assetWordsFile):
    print(dictLine)
    dictLine=dictLine.replace('"','')
    print(dictLine)
    arrPumpWords=dictLine.split(",")
    break


#read the unigram file and create a dictioanry
unigramFile=prefixpath+"unigramWO.csv"
dictUnigrams={}
for ll in open(unigramFile):
    llArr=ll.split(",")
    key=llArr[0].replace('"','')
    val=int(llArr[1].replace('\n',''))
    dictUnigrams[key]=val

print(dictUnigrams)


#read the bigrams with their frequecny
bigramFile=prefixpath+"bigrams.csv"
dictBigrams={}
for ll in open(bigramFile):
    llArr=ll.split(",")
    key = llArr[0].replace('"', '')
    val=int(llArr[1].replace('\n',''))
    dictBigrams[key]=val

print(dictBigrams)


#read the trigrams with their frequecy
trigramFile=prefixpath+"trigramsWO.csv"
dictTrigrams={}
for ll in open(trigramFile):
    llArr=ll.split(",")
    key=llArr[0].replace('"','')
    val=int(llArr[1].replace('\n',''))
    dictTrigrams[key]=val

print(dictTrigrams)




#read the stopwrods
stopwordsFile=prefixpath+"WOStopwords.csv"
arrStopwords=[]
for ll in open(stopwordsFile):
    llArr=ll.split(",")
    val=llArr[1].replace('\n','')
    val=val.replace('"','')
    arrStopwords.append(val)

print(arrStopwords)
stops=set(arrStopwords)
print(stops)

import mysql.connector

'''
Use the database to fetch the Workorders that you want to analyze. Currently they are based on the equipment type
'''
cnx = mysql.connector.connect(user='root', password='root',host='localhost',port='3306',database='assetanswers')

result=[]
try:
   cursor = cnx.cursor()
   cursor.execute("""SELECT DISTINCT(WH_ORIG_RQST_DESC_C) from workorders_assets where EQ_EQ_CLASS_C='Mixer'""")
   result = cursor.fetchall()
   print(result)
finally:
    cnx.close()


#checkWhichMatches not being used anymore
def checkWhichMatches(testArray,wordArray):
   matchset=[]
   unmatch=[]
   for a in testArray:
       if(len(a)==0):
           continue
       if a in wordArray:
           matchset.append(a)
       else:
           unmatch.append(a)
   return(matchset,unmatch)

'''
Get the various bigrams that can be formed from the text..match them with the dictionary
'''
def getBigramsDistributionFromText(txt):
    bigrm = list(bigrams(txt.split()))
    # print(bigrm)
    bigramWords = ', '.join(' '.join((a, b)) for a, b in bigrm)

    dictResBi = {}
    for bi in bigramWords.split(","):
        bi = bi.lstrip()
        bi = bi.rstrip()
        if bi in dictBigrams:
            # print(bi,dictBigrams[bi])
            dictResBi[bi]=dictBigrams[bi]
        # else:
        #     # print("NA")
        #     dictResBi[bi]= 0

    return (sorted(dictResBi.items(), key=lambda x: x[1], reverse=True))

'''
Get all the unigrams...these are the words matching the dictioary
'''
def getUnigramDistributionFromText(txt):
    arrUnigramWords=txt.split(" ")
    dictResUni={}
    for uni in arrUnigramWords:
        uni = uni.lstrip()
        uni = uni.rstrip()
        if uni in dictUnigrams:
            # print(bi,dictBigrams[bi])
            dictResUni[uni]=dictUnigrams[uni]
        # else:
        #     # print("NA")
        #     dictResUni[uni]= 0
    return(sorted(dictResUni.items(), key=lambda x: x[1], reverse=True))

    #return(arrUni)

'''
Get the trigrams ..match them with the dictionary. The chances of having trigrams is very low as the sentence are short
'''
def getTrigramsDistributionFromText(txt):
    trigrm = list(trigrams(txt.split()))
    # print(bigrm)
    trigramWords = ', '.join(' '.join((a, b, c)) for a, b,c in trigrm)

    dictResTri = {}
    for tri in trigramWords.split(","):
        tri = tri.lstrip()
        tri = tri.rstrip()
        if tri in dictTrigrams:
            # print(bi,dictBigrams[bi])
            dictResTri[tri]=dictTrigrams[tri]
        # else:
        #     # print("NA")
        #     dictResBi[bi]= 0

    return (sorted(dictResTri.items(), key=lambda x: x[1], reverse=True))

'''
You will perform tyhe basic cleanup ...numbers, punctuations, stopwords...Note that the stopwords should be quite
similar to what were used at the time of creating the dictionary otherwise some of the bigrams will not match
'''
def getCleanedUpTextString(txt):
    txt = txt.lower()
    original = txt
    txt = re.sub("[^a-zA-Z]", " ", txt)
    txtWithPunctuationRemoved = txt
    arrtxt = txt.split(" ")
    # remove stop words
    # stops = set(stopwords.words("english"))
    words = [w for w in arrtxt if not w in stops]
    words = [w for w in words if len(w) > 1]
    finalString = ' '.join(words)
    return(original,finalString)




'''
Main function:
Read the workorders from the database
find out the unigrams, bigrams,trigrams
and store them into a file for further analysis

'''
import json
fp = open(prefixpath+"cognitiveOutput.txt", "w")
outputResult=[]

for rr in result[1:100]:
    originalTxt, cleanedUpTxt=getCleanedUpTextString(rr[0])

    #stemming
   # stemmed = [stemmer.stem(ww) for ww in words]
   # stemmed=' '.join(stemmed)


    #words=txt.split(" ")
#    print(words)
    #matchUni,unmatchUni=checkWhichMatches(words,arrPumpWords)

    #get the unigrams and choose the top 3
    uniStr=getUnigramDistributionFromText(cleanedUpTxt)
    # uniLen=len(uniStr)
    # if(uniLen>5):
    #     uniStr=uniStr[0:5]

    #get the bigrams and choose the top 3
    biStr=getBigramsDistributionFromText(cleanedUpTxt)
    biLen=len(biStr)
    # if(biLen>5):
    #     biStr=biStr[0:5]
    #sorted(biStr.items(), key=lambda x: x[1], reverse=True)

    #get the trigrams and choose the top 3
    triStr = getTrigramsDistributionFromText(cleanedUpTxt)
    triLen = len(triStr)
    if (triLen > 5):
        triStr = triStr[0:5]

    stringifiedTrigram = ', '.join(tt[0] for tt in triStr)
    stringifiedBigram=', '.join(bb[0] for bb in biStr)
    stringifiedUnigram=', '.join(uu[0] for uu in uniStr)

    print(originalTxt, "....",stringifiedTrigram,"...",stringifiedBigram, "...", stringifiedUnigram)
    outputResult.append({"original":originalTxt,"trigrams":stringifiedTrigram,"bigrams":stringifiedBigram,"unigrams":stringifiedUnigram})
    # fp.write(originalTxt)
    # fp.write("....")
    # fp.write(stringifiedTrigram)
    # fp.write("...")
    # fp.write(stringifiedBigram)
    # fp.write("...")
    # fp.write(stringifiedUnigram)
    # fp.write("\n")

    #pp = pd.DataFrame(list(zip(unique, counts)), columns=['num of occurence', 'freq'])

    #print(original)

fp.write(json.dumps(outputResult))
fp.close()





''''EXTRA NOT TO BE CONSIDERED'''''
###############################
#result[1][0]



bigrm = list(bigrams(result[2][0].split()))

#print(bigrm)
print(', '.join(' '.join((a, b)) for a, b in bigrm))


print(dictBigrams)



#print(arrPumpWords)

#type(arrPumpWords)

#stemmer.stem('inspection of the woks')

originalTxt, cleanedUpTxt=getCleanedUpTextString(result[13][0])


tt=getTrigramsDistributionFromText(cleanedUpTxt)

list(trigrams(cleanedUpTxt.split()))
#stemming
# stemmed = [stemmer.stem(ww) for ww in words]
# stemmed=' '.join(stemmed)



#words=txt.split(" ")
#    print(words)
#matchUni,unmatchUni=checkWhichMatches(words,arrPumpWords)

uniStr=getUnigramDistributionFromText(cleanedUpTxt)
uniLen=len(uniStr)
print(uniLen)
if(uniLen>3):
    uniStr=uniStr[0:3]

print(uniStr)
#get the bigrams
biStr=getBigramsDistributionFromText(cleanedUpTxt)
biLen=len(biStr)
if(biLen>3):
    biStr=biStr[0:2]
#sorted(biStr.items(), key=lambda x: x[1], reverse=True)
print(biStr[0][0])

print(originalTxt, "....", uniStr, "...", biStr)
    #pp = pd.DataFrame(list(zip(unique, counts)), columns=['n

print(dictTrigrams)