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


#
# arrPumpWords=[]
#
# assetWordsFile=prefixpath+"pumpKeywords1.csv"
# for dictLine in open(assetWordsFile):
#     print(dictLine)
#     dictLine=dictLine.replace('"','')
#     print(dictLine)
#     arrPumpWords=dictLine.split(",")
#     break


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
def getData(equipmentType):

    cnx = mysql.connector.connect(user='root', password='root',host='localhost',port='3306',database='assetanswers')

    result=[]
    try:
       cursor = cnx.cursor()
       sqlQuery="SELECT DISTINCT(WH_ORIG_RQST_DESC_C) from workorders_assets where EQ_EQ_CLASS_C='%s'" %(equipmentType)
       cursor.execute(sqlQuery)
       result = cursor.fetchall()
       print(result)
    finally:
        cnx.close()

        return(result)


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


def getTotalDistribution(ngramArray,finalCountDict):
    for nn in ngramArray:
        if nn[0] in countNgrams:
            finalCountDict[nn[0]] += float(1)
        else:
            finalCountDict[nn[0]] = float(1)

    return(finalCountDict)


def drawFreqDistribution(sortedCountNGrams):
    import numpy as np
    import matplotlib.pyplot as plt
    labelArr = []
    freqArr = []

    for obj in sortedCountNGrams[0:30]:
        labelArr.append(obj[0])
        freqArr.append(int(obj[1]))

    N = len(labelArr)
    ind = np.arange(N)  # the x locations for the groups
    width = 0.50  # the width of the bars

    fig, ax = plt.subplots()
    winners = ax.barh(ind, freqArr, width, color='green')
    ax.set_yticks(ind + width)
    ax.set_yticklabels(labels=labelArr)


'''
We need to make sure that if the constituent words of bigram are present in unigram then unigram will be discarded
similarly if the words in trigram are present in the unigram then they are to be removed
also if bigram is a subset of trigram then it will not be counted

'''
# def findEffectiveNgrams(trigramArr,bigramArr,unigramArr):
#     ##do something


def findEffectiveBigrams(bigramArr,TrigramArr):
    finalbb = []
    for bb in bigramArr:
        initList = set(bb[0].split(' '))
        # print("bi:", initList)
        flag = 1
        for tt in TrigramArr:
            trList = set(tt[0].split(' '))
            # print("tri:", tt[0])
            diffLst = (initList.difference(trList))
            # print(diffLst, len(diffLst))
            if (len(diffLst) == 0):
                # print("sub:", bb)
                flag = 0
                # break
                # print(bb[0])
        if (flag == 1):
            finalbb.append(bb)

    return(finalbb)

'''
Main function:
Read the workorders from the database
find out the unigrams, bigrams,trigrams
and store them into a file for further analysis

'''

import json
fp = open(prefixpath+"cognitiveOutput.txt", "w")
outputResult=[]
result=getData("Pump")
#count the number of times each word has come...this way we can give a distribution and may be that is helpful
countNgrams={}
for rr in result:
    originalTxt, cleanedUpTxt=getCleanedUpTextString(rr[0])

    #stemming
   # stemmed = [stemmer.stem(ww) for ww in words]
   # stemmed=' '.join(stemmed)
    #get the unigrams and choose the top 3
    unigramWords=[]
    constituentNGramWords=[]

    uniArr=getUnigramDistributionFromText(cleanedUpTxt)
    print(uniArr)
    for uu in uniArr:
        unigramWords.append(uu[0])

    #countNgrams=getTotalDistribution(uniArr,countNgrams)

    #get the bigrams and choose the top 3
    biArr=getBigramsDistributionFromText(cleanedUpTxt)
    biLen=len(biArr)
    for bb in biArr:
        # bigramWords.append(bb[0])
        for word in bb[0].split(' '):
            constituentNGramWords.append(word)


    #countNgrams = getTotalDistribution(biArr, countNgrams)

    #get the trigrams and choose the top 3
    triArr = getTrigramsDistributionFromText(cleanedUpTxt)
    triLen = len(triArr)
    for tt in triArr:
        # trigramWords.append(tt[0])
        for word in tt[0].split(' '):
            constituentNGramWords.append(word)


    finalbb=findEffectiveBigrams(biArr,triArr)

    countNgrams = getTotalDistribution(finalbb, countNgrams)
    countNgrams = getTotalDistribution(triArr, countNgrams)

    #before adding the number of unigrams we need to findout how many are not covered by the bigrams,trigrams
    finUnigrams = list(set(unigramWords).difference(constituentNGramWords))
    #now add these unigrams frequeccy
    for uu in uniArr:
        if uu[0] in finUnigrams:
            if uu[0] in countNgrams:
                countNgrams[uu[0]] += float(1)
            else:
                countNgrams[uu[0]] = float(1)

    stringifiedTrigram = ', '.join(tt[0] for tt in triArr)
    stringifiedBigram=', '.join(bb[0] for bb in finalbb)
    stringifiedUnigram=', '.join(uu for uu in finUnigrams)


    #effectiveNgrams=findEffectiveNgrams(triArr,biArr,uniArr)



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
sortedCountNGrams=sorted(countNgrams.items(), key=lambda x: x[1], reverse=True)
print(sortedCountNGrams[0:50])

fp.write(json.dumps(outputResult))
fp.close()

#print(sortedCountNGrams[0:100])
drawFreqDistribution(sortedCountNGrams)


# def autolabel(rects):
#     # attach some text labels
#     for rect in rects:
#         height = rect.get_height()
#         hcap = "$"+str(height)+"M"
#         ax.text(rect.get_x()+rect.get_width()/2., height, hcap,ha='center', va='bottom', rotation="vertical")



#autolabel(winners)






'''
SOme thoughts

If we just create a frequency map of the various keywords that we are getting on mixer it gives a pretty good picture of
what are the key issues that are going on....I think I will try to integrate the word cloud creation code so that generate
the cloud for each equipment type




We need to check whihc words are coming after not...they may be interesting
'''




''''EXTRA NOT TO BE CONSIDERED'''''
###############################
#result[1][0]


#for rr in result[1:10]

originalTxt, cleanedUpTxt=getCleanedUpTextString(result[7][0])

unigramWords=[]
bigramWords=[]
trigramWords=[]
countNgrams={}

#get the unigrams and choose the top 3
uniArr=getUnigramDistributionFromText(cleanedUpTxt)
for uu in uniArr:
    unigramWords.append(uu[0])

constituentUnigramsWords=[]


countNgrams = getTotalDistribution(uniArr, countNgrams)

biArr=getBigramsDistributionFromText(cleanedUpTxt)
for bb in biArr:
    #bigramWords.append(bb[0])
    for word in bb[0].split(' '):
        constituentUnigramsWords.append(word)
countNgrams = getTotalDistribution(biArr, countNgrams)

triArr = getTrigramsDistributionFromText(cleanedUpTxt)
for tt in triArr:
    #trigramWords.append(tt[0])
    for word in tt[0].split(' '):
        constituentUnigramsWords.append(word)

finalbb=[]
for bb in biArr:
    initList=set(bb[0].split(' '))
    print("bi:",initList)
    flag=1
    for tt in triArr:
        trList=set(tt[0].split(' '))
        print("tri:",tt[0])
        diffLst=(initList.difference(trList))
        print(diffLst,len(diffLst))
        if(len(diffLst)==0):
            print("sub:",bb)
            flag=0
            #break
        #print(bb[0])
    if(flag==1):
        finalbb.append(bb)

print(triArr)
print(finalbb)


countNgrams = getTotalDistribution(triArr, countNgrams)

print(countNgrams)
print(uniArr)
print(constituentWords)
print(set(constituentWords))
print(set(unigramWords))
#these are the final unigrams to be considered
finUnigrams=list(set(unigramWords).difference(constituentWords))
print(list(finUnigrams))
print(biArr)
print(triArr)
for uu in uniArr:
    if uu[0] in finUnigrams:
        if uu[0] in countNgrams:
            countNgrams[uu[0]] += float(1)
        else:
            countNgrams[uu[0]] = float(1)


#getTotalDistribution(triArr, finalCountDict)