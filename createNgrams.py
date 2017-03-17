
#
# This script is used to find the unigrams in the text using the tf-idf scoring method
# Makes use of minimal stopwords
#  the assets are also identified in the text and the muliple word asset is replaced with a _ joined word
# all the data from the trienergy is used for finding the unigrams

from nltk.tokenize import WordPunctTokenizer
from nltk.collocations import BigramCollocationFinder
from nltk.collocations import TrigramCollocationFinder
from nltk.metrics import TrigramAssocMeasures
from nltk.metrics import BigramAssocMeasures


import re
import mysql.connector

prefixpath="/Users/305015992/pythonProjects/assetAnswer/"

def initialize(prefixpath):
    cnx = mysql.connector.connect(user='root', password='root', host='localhost', port='3306', database='assetanswers')

    result = []
    try:
        cursor = cnx.cursor()
        sqlQuery = "SELECT WO_Desc FROM TrienergyWOforNgrams"
        print(sqlQuery)
        cursor.execute(sqlQuery)
        result = cursor.fetchall()
        print(result)
    finally:
        cnx.close()


    return result

cases=initialize(prefixpath)

stopwordsFile = prefixpath + "WOStopwords.csv"
arrStopwords = []
for ll in open(stopwordsFile):
    # print(ll)
    llArr = ll.split(",")
    # print(llArr)
    if (len(llArr) > 1):
        val = llArr[1].replace('\n', '')
        val = val.replace('"', '')
        print(val)
        arrStopwords.append(val)

print(arrStopwords)
stopwordList = set(arrStopwords)

import re
import numpy as np

def getCleanedUpTextString(txt,stops):
    txt = txt.lower()
    original = txt
    #first remove the code FWA-P-201A in the strings and may be we can store the stuff as well
    txt = re.sub('[a-zA-Z0-9]+-[0-9A-Za-z]+-[0-9a-zA-Z]+', '', txt)

    #remove every thing other than alphabets
    txt = re.sub("[^a-zA-Z]", " ", txt)

    #txt=getReplacedWithAssets(txt)


    #replace check/record to check
    #txt=txt.replace("check record","check")
    txtWithPunctuationRemoved = txt
    arrtxt = txt.split(" ")
    # remove stop words
    # stops = set(stopwords.words("english"))
    words = [w for w in arrtxt if not w in stops]
    words = [w for w in words if len(w) > 1]
    finalString = ' '.join(words)
    return(original,finalString)



cleanedUpCases=[]
for rr in cases:
    original,cleanup=getCleanedUpTextString(rr[0],stopwordList)
    cleanedUpCases.append(cleanup)



#now we want to create the dictionary based on the ngram analysis
from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer(min_df=0.0001,stop_words=stopwordList,strip_accents='unicode',binary=False)
rawDtm = count_vect.fit_transform(cleanedUpCases)
#maximum is 4997
#rawDtm = vectorizer.fit_transform(cleanedUpCases[0:4900])
print("Data dimensions: {}".format(rawDtm.shape))

vocab=count_vect.get_feature_names()
print(len(vocab))

print(vocab)

countDtm = rawDtm.toarray()
countDtm=np.array(countDtm)

freqsum=np.sum(countDtm,axis=0)

np.amax(freqsum)
np.amin(freqsum)

freqDict={}
for idx,v in enumerate(vocab):
    #freqDict.append({'word':v,'count':freqsum[idx]})
    freqDict[v]=freqsum[idx]


print(freqDict)


'''creating the sorted list of the freqdict'''

# sorting the dictionary using the value ..http://stackoverflow.com/questions/613183/sort-a-python-dictionary-by-value
#there are multiple ways i am using the itemgetter way
sortedFreqDict=[]
for w in sorted(freqDict, key=freqDict.get, reverse=True):
    print(w, freqDict[w])
    #sortedFreqDict[]

sorted(freqDict.items(), key=lambda x: x[1])

from operator import itemgetter
sortedFreqDict=sorted(freqDict.items(), key=itemgetter(1),reverse=True)

#this gives the list of the unigrams
print(sortedFreqDict)

def getFreqOfKeyword(freqDict,keyword):
    for key in freqDict:
        if(key==keyword):
            return(freqDict[key])

print(getFreqOfKeyword(freqDict,"bearing"))
#
#
# def getKeywordBasedOnFrequency(freqDict,freq):
#     for key in freqDict:
#         if (freqDict[key]==freq):
#             return (key)
#
# print(getKeywordBasedOnFrequency(freqDict,81))

#unigrams:
#list of unigrams
# getKeywordBasedOnFrequency(np.amin(freqsum))
# print(vocab)

fu=open("/Users/305015992/pythonProjects/assetAnswer/trienergyNgram/unigrams_trienergy_R.txt","w")
for uu in sortedFreqDict:
    uni=uu[0]+","+str(uu[1])
    fu.write(uni)
    fu.write("\n")
fu.close()


##############################################
#this is bigram extraction based on predicting what cmones next P(W2|W1)
#for now we are not using this but we need to explore this more...
#I guess one of the imporvement required is dont restrict to 5 words per bigram
#also the next word should also be a unigram

def findNextCountsForWords(arrWords,countNextWords,val):
    for ii in arrWords:
        if(ii in countNextWords):
            countNextWords[ii]+=float(1)
        else:
            countNextWords[ii]=float(1)

    return(countNextWords)

#now we need to find the bigrams
def getPossibleNextWordsAfterPrticularWord(result,matchingWord,val,stops):
    countNextWords={}
    resultStringMatches={}
    countPrevWords={}
    for rr in result:
        #originalTxt, cleanedUpTxt = getCleanedUpTextString(rr[0])
        originalTxt=rr[0].lower()
        originalTxt = re.sub('[a-zA-Z0-9]+-[0-9A-Za-z]+-[0-9a-zA-Z]+', '', originalTxt)
        #
        # # remove every thing other than alphabets
        originalTxt = re.sub("[^a-zA-Z]",' ', originalTxt)
        #remove extra two spaces
        originalTxt=originalTxt.replace('  ',' ')
        #convert the asset names
        #originalTxt=getReplacedWithAssets(originalTxt)


        matchString=matchingWord
        #handle the case when check/record is present
        words=originalTxt.partition(matchString)
        #print(words)
        if(len(words[2])>1):
            wordsafter=words[2].split()
            if(len(wordsafter)>0):
                if (wordsafter[0] not in stops):
                    countNextWords=findNextCountsForWords([wordsafter[0]],countNextWords,val)

    outputNextWords=sorted(countNextWords.items(), key=lambda x: x[1], reverse=True)
    return(outputNextWords)


len(cases)


unigrams=[]

for uu in sortedFreqDict:
    unigrams.append(uu[0])



fp=open("bigram_temp1","a")
bigrams=[]
count=0
for uu in sortedFreqDict[501:1000]:
    count=count+1
    print(count)
    print(uu[0],uu[1])
    testUnigram=uu[0]
    #val=getFreqOfKeyword(freqDict,testUnigram)
    output=getPossibleNextWordsAfterPrticularWord(cases,testUnigram,uu[1],stopwordList)
    # print(output)
    # if(len(output)>5):
    #     print(output[0:5])
    # else:
    #     print(output)

    # if(len(output)>5):
    #     output=output[0:5]

    for oo in output:
        if ((oo[0] in unigrams) and int(oo[1])>10):
            bigram=uu[0]+" "+oo[0]+","+str(oo[1])
            fp.write(bigram)
            fp.write("\n")
            print(bigram)
            bigrams.append(bigram)


print(bigrams)
fp.close()
#print(len(cases))
#getPossibleNextWordsAfterPrticularWord(cases,"check",12812,stopwordList)

#     fu.write(uu[0]+","+uu[1])
#     fu.write("\n")
# fu.close()


#we need to try out the one given by bigram creation API in pythoon

'''
def get_bigrams(myString):
    tokenizer = WordPunctTokenizer()
    tokens = tokenizer.tokenize(myString)
    cleanedTokens=[x for x in tokens if x.lower() not in stopwordList]
    # print(tokens)
    #stemmer = PorterStemmer()
    bigram_finder = BigramCollocationFinder.from_words(cleanedTokens)
    #bigrams = bigram_finder.nbest(BigramAssocMeasures.chi_sq, 5)
    #print(bigrams)

    # for bigram_tuple in bigrams:
    #     x = "%s %s" % bigram_tuple
    #     tokens.append(x)

    return bigram_finder.ngram_fd.items()
    # for k, v in bigram_finder.ngram_fd.items():

'''




fb=open("/Users/305015992/pythonProjects/assetAnswer/trienergyNgram/bigrams_new.txt","w")
for bb in bigrams:
    fb.write(bb)
    fb.write("\n")
fb.close()







