'''
This script is to find out the issues disrtibution for a particular site and equipment.
It takes as an input the unigram list, bigram list and trigram list
'''
import numpy as np
from nltk.util import bigrams
from nltk.util import trigrams
import csv
from nltk.corpus import stopwords
import re
import mysql.connector

from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()

import pandas as pd


prefixpath="/Users/305015992/pythonProjects/assetAnswer/"

########## READ FILES TO CONFIGURE NGRAMS ###################

'''
read the unigram file and create a dictioanry
'''
unigramFile=prefixpath+"trienergyNgram/unigrams_trienergy_R.txt"
dictUnigrams={}
for ll in open(unigramFile):
    llArr=ll.split(",")
    key=llArr[0].replace('"','')
    val=int(llArr[1].replace('\n',''))
    dictUnigrams[key]=val

print(dictUnigrams)

'''
this is a list of unigrams which we feel that shoul dbe removed. like stop words but they are not actually stop words.
We will make use of the list just before publishing the final words
'''
undesiredWordsFile=prefixpath+"trienergyNgram/UndesiredWords.csv"
arrUndesiredWords=[]
for ll in open(undesiredWordsFile):
    #print(ll)
    llArr=ll.split(",")
    #print(llArr)
    if(len(llArr)>1):
        val=llArr[1].replace('\n','')
        val=val.replace('"','')
        print(val)
        arrUndesiredWords.append(val)

print(arrUndesiredWords)

'''
read the bigrams with their frequecny
'''
bigramFile=prefixpath+"trienergyNgram/bigrams_trienergy_R.csv"
dictBigrams={}
for ll in open(bigramFile):
    llArr=ll.split(",")
    key = llArr[0].replace('"', '')
    val=int(llArr[1].replace('\n',''))
    dictBigrams[key]=val

print(dictBigrams)


'''
read the trigrams with their frequecy
'''
trigramFile=prefixpath+"trienergyNgram/trigrams_trienergy_R.csv"
dictTrigrams={}
for ll in open(trigramFile):
    llArr=ll.split(",")
    key=llArr[0].replace('"','')
    val=int(llArr[1].replace('\n',''))
    dictTrigrams[key]=val

print(dictTrigrams)

'''
read the stopwrods
'''
stopwordsFile=prefixpath+"WOStopwords.csv"
arrStopwords=[]
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
stops=set(arrStopwords)
print(stops)

'''
read the asset file
'''
fname=prefixpath+"trienergyNgram/assetNames.txt"
with open(fname, 'r') as myfile:
    data=myfile.read()

assets={}
for dd in data.split("\n"):
    dd=dd.lower()
    assets[dd]=dd.replace(' ','_')

assets




'''
getData: Use the database to fetch the Workorders that you want to analyze. in this case the equipmentType and sitename is not being used
as we are strictly doing it for trienergy pump...but we should be able to change this or may be get the entire sql query
in the parameter
'''
def getData(equipmentType,siteName,isDistinct=True):

    cnx = mysql.connector.connect(user='root', password='root',host='localhost',port='3306',database='assetanswers')

    result=[]
    try:
       cursor = cnx.cursor()
       sqlQuery = "SELECT WO_Long_Desc FROM TrienergyWO LIMIT 100"
       print(sqlQuery)
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
           matchset.append(a)
           unmatch.append(a)
   return(matchset,unmatch)


'''
getBigramsDistributionFromText: Get the various bigrams that can be formed from the text..match them with the dictionary
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
getUnigramDistributionFromText: Get all the unigrams...these are the words matching the dictioary
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
getTrigramsDistributionFromText: Get the trigrams ..match them with the dictionary. The chances of having trigrams is very low as the sentence are short
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


def getReplacedWithAssets(txt):
    for aa in assets:
        txt=txt.replace(aa,assets[aa])

    return(txt)




'''
getCleanedUpTextString: You will perform tyhe basic cleanup ...numbers, punctuations, stopwords...Note that the stopwords should be quite
similar to what were used at the time of creating the dictionary otherwise some of the bigrams will not match
'''
def getCleanedUpTextString(txt,replaceAssets=True,removeStopwords=True):
    txt = txt.lower()
    original = txt
    #first remove the code FWA-P-201A in the strings and may be we can store the stuff as well
    txt = re.sub('[a-zA-Z0-9]+-[0-9A-Za-z]+-[0-9a-zA-Z]+', '', txt)

    #remove every thing other than alphabets
    txt = re.sub("[^a-zA-Z]", " ", txt)

    if(replaceAssets==True):
        txt=getReplacedWithAssets(txt)


    #replace check/record to check
    #txt=txt.replace("check record","check")
    #txtWithPunctuationRemoved = txt
    if(removeStopwords==True):
        arrtxt = txt.split(" ")
        # remove stop words
        # stops = set(stopwords.words("english"))
        words = [w for w in arrtxt if not w in stops]
        words = [w for w in words if len(w) > 1]
        finalString = ' '.join(words)
    else:
        finalString=txt
    return(original,finalString)



''''
getTotalDistribution: Get the final count of the various Ngrams in the corpus. So what is the final frequency for the bigram words,, trigram words
'''
def getTotalDistribution(ngramArray,finalCountDict):
    for nn in ngramArray:
        if nn[0] in finalCountDict:
            finalCountDict[nn[0]] += float(1)
        else:
            finalCountDict[nn[0]] = float(1)

    return(finalCountDict)


'''
drawFreqDistribution: Draws the frequecny distribution in the form of a bar graph
'''
def drawFreqDistribution(sortedCountNGrams,N=30):
    import numpy as np
    import matplotlib.pyplot as plt
    labelArr = []
    freqArr = []

    for obj in sortedCountNGrams[0:N]:
        labelArr.append(obj[0])
        freqArr.append(int(obj[1]))

    N = len(labelArr)
    ind = np.arange(N)  # the x locations for the groups
    width = 0.50  # the width of the bars

    fig, ax = plt.subplots()
    winners = ax.barh(ind, freqArr, width, color='green')
    ax.set_yticks(ind + width)
    ax.set_yticklabels(labels=labelArr)

    for rect in winners:
        # print(rect)
        width_rect = int(rect.get_width())
        # print(width_rect)
        hcap = str(width_rect)
        ax.text(width_rect + 2, rect.get_y() - 0.25, hcap, ha='center', va='bottom', rotation="horizontal")

    return(labelArr)


'''
findEffectiveBigrams: if the words of bigram are present in the trigram then the bigram needs to be discarded
'''
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



#we need to work on this ..
def removeOverlappingTrigramsWithHighFreq(TrigramArr):
    print(TrigramArr)
    finaltri=[]
    for tri in TrigramArr:
        initList=tri[0].split(' ')
        flag = 1
        for tt in TrigramArr:
            trList=tt[0].split(' ')
            if((initList[1]==trList[0] and initList[2]==trList[1]) or (initList[2]==trList[0])):
                print(trList,initList)
                if(dictTrigrams[tt[0]]>=dictTrigrams[tri[0]]):
                    finaltri.append(tt)
    return(set(finaltri))


'''
removeUndesiredUnigrams: This function aims at removing some of the undesired unigrams like change, clear, new etc
these words might be used in the bigrams so they are not stop words, but as unigrams they are noise
'''
def removeUndesiredUnigrams(unigramArr):
    finArr=[]
    for uu in unigramArr:
        if uu in arrUndesiredWords:
            continue
        else:
            finArr.append(uu)
    return(finArr)


'''
Main function:
Read the workorders from the database
find out the unigrams, bigrams,trigrams
and store them into a file for further analysis

'''

import json
fp = open(prefixpath+"cognitiveOutput2.txt", "w")
outputResult=[]
#give the eqquipment class and the site
result=getData("Pump","Trienergy",isDistinct=False)

#count the number of times each word has come...this way we can give a distribution and may be that is helpful
countNgrams={}
for rr in result:
    originalTxt, cleanedUpTxt=getCleanedUpTextString(rr[0],True,False)

    #stemming
   # stemmed = [stemmer.stem(ww) for ww in words]
   # stemmed=' '.join(stemmed)
    #get the unigrams and choose the top 3
    unigramWords=[]
    constituentNGramWords=[]

    uniArr=getUnigramDistributionFromText(cleanedUpTxt)
    #print(uniArr)
    for uu in uniArr:
        unigramWords.append(uu[0])
        # if(uu[0]=="change"):
        #     print("change",originalTxt)
        #     print("\n")

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


    #finalbb=biArr
    countNgrams = getTotalDistribution(finalbb, countNgrams)



    countNgrams = getTotalDistribution(triArr, countNgrams)

    #before adding the number of unigrams we need to findout how many are not covered by the bigrams,trigrams
    finUnigrams = list(set(unigramWords).difference(constituentNGramWords))
    finUnigrams=removeUndesiredUnigrams(finUnigrams)
    #now add these unigrams frequeccy
    for uu in uniArr:
        if uu[0] in finUnigrams:
            if uu[0] in countNgrams:
                countNgrams[uu[0]] += float(1)
            else:
                countNgrams[uu[0]] = float(1)



    #in case of trigram remove the trigram which overlaps and has a lower prob of occurence

    #newTriArr=removeOverlappingTrigramsWithHighFreq(triArr)




    stringifiedTrigram = ', '.join(tt[0].replace(' ','_') for tt in triArr)
    #stringifiedNewTrigram = ', '.join(tt[0].replace(' ', '_') for tt in newTriArr)
    stringifiedBigram=', '.join(bb[0].replace(' ','_') for bb in finalbb)
    stringifiedUnigram=', '.join(uu for uu in finUnigrams)

    print(originalTxt, "~",stringifiedTrigram,"~",stringifiedBigram, "~", stringifiedUnigram)
    outputResult.append({"original":originalTxt,"trigrams":stringifiedTrigram,"bigrams":stringifiedBigram,"unigrams":stringifiedUnigram})
    #printString=originalTxt+"..Tri="+stringifiedTrigram+"...Bi=",stringifiedBigram+"...Uni="+ stringifiedUnigram

    #fp.write(printString)
    #fp.write("\n")
    #print(original)

print(countNgrams)
sortedCountNGrams=sorted(countNgrams.items(), key=lambda x: x[1], reverse=True)
print(sortedCountNGrams[0:50])
#print(sortedCountNGrams[0:100])
toplabels=drawFreqDistribution(sortedCountNGrams,N=100)
fp.write(json.dumps(outputResult))
fp.write("..........\n")
fp.write(json.dumps(toplabels))
fp.close()

for tt in sortedCountNGrams:
    if(tt[1]>10):
        print(tt[0],',',tt[1])




##test
def findNextCounts(arrWords,countNextWords,val):
    for ii in arrWords:
        if(ii in countNextWords):
            countNextWords[ii]+=float(1)/float(val)
        else:
            countNextWords[ii]=float(1)/float(val)

    return(countNextWords)


def findPrevCounts(arrWords,countPrevWords,val):
    for ii in arrWords:
        if(ii in countPrevWords):
            countPrevWords[ii]+=float(1)/float(val)
        else:
            countPrevWords[ii]=float(1)/float(val)

    return(countPrevWords)


def findNextCountsForWords(arrWords,countNextWords):
    for ii in arrWords:
        if(ii in countNextWords):
            countNextWords[ii]+=float(1)
        else:
            countNextWords[ii]=float(1)

    return(countNextWords)


def getPossibleNextWords(result,initialString,initalVal):
    countNextWords={}
    countPrevWords={}
    for rr in result:
        originalTxt, cleanedUpTxt = getCleanedUpTextString(rr[0],False,False)
        originalTxt = re.sub('[a-zA-Z0-9]+-[0-9A-Za-z]+-[0-9a-zA-Z]+', '', originalTxt)
        #
        # # remove every thing other than alphabets
        originalTxt = re.sub("[^a-zA-Z]",' ', originalTxt)
        #remove extra two spaces
        originalTxt=originalTxt.replace('  ',' ')
        #convert the asset names
        originalTxt=getReplacedWithAssets(originalTxt)


        matchString=initialString
        #handle the case when check/record is present
        words=cleanedUpTxt.partition(matchString)
        #print(words)
        if(len(words[2])>1):
            wordsafter=words[2].split()
            #print("wordsafter length",len(wordsafter))
            if (len(wordsafter) > 2):
                # check 3 words first, second, first&second
                countNextWords = findNextCounts([wordsafter[0], wordsafter[1], wordsafter[0] + ' ' + wordsafter[1],
                                                 wordsafter[0] + ' ' + wordsafter[1]+' '+wordsafter[2]],
                                                countNextWords, initalVal)

            if(len(wordsafter)>1 and len(wordsafter)<3):
                #check 3 words first, second, first&second
                countNextWords=findNextCounts([wordsafter[0],wordsafter[1],wordsafter[0]+' '+wordsafter[1]],countNextWords,initalVal)
            if(len(wordsafter)==1):
                countNextWords=findNextCounts([wordsafter[0]],countNextWords,initalVal)

        # if(len(words[0])>1):
        #     wordsbefore=words[0].split()
        #     if (len(wordsbefore) >1):
        #         # check 3 words first, second, first&second
        #         countPrevWords = findPrevCounts([wordsbefore[0], wordsbefore[1], wordsbefore[0] + ' ' + wordsbefore[1]],
        #                                          countPrevWords, initalVal)
        #     if (len(wordsbefore) == 1):
        #         countPrevWords = findPrevCounts([wordsbefore[0]], countPrevWords, initalVal)
        #     print(wordsbefore)
    outputNextWords=sorted(countNextWords.items(), key=lambda x: x[1], reverse=True)
    # outputPrevWords=sorted(countPrevWords.items(),key=lambda x: x[1], reverse=True)

    return(outputNextWords)


#getPossibleNextWords(result,"check record battery",float(188))

def getPossibleNextWordsAfterPrticularWord(result,matchingWord):
    countNextWords={}
    resultStringMatches={}
    countPrevWords={}
    for rr in result:
        originalTxt, cleanedUpTxt = getCleanedUpTextString(rr[0])
        originalTxt = re.sub('[a-zA-Z0-9]+-[0-9A-Za-z]+-[0-9a-zA-Z]+', '', originalTxt)
        #
        # # remove every thing other than alphabets
        originalTxt = re.sub("[^a-zA-Z]",' ', originalTxt)
        #remove extra two spaces
        originalTxt=originalTxt.replace('  ',' ')
        #convert the asset names
        originalTxt=getReplacedWithAssets(originalTxt)


        matchString=matchingWord
        #handle the case when check/record is present
        words=originalTxt.partition(matchString)
        #print(words)
        if(len(words[2])>1):
            wordsafter=words[2].split()
            #print("wordsafter length",len(wordsafter))
            if (len(wordsafter) > 2):
                # check 3 words first, second, first&second
                countNextWords = findNextCountsForWords([wordsafter[0], wordsafter[1], wordsafter[0] + ' ' + wordsafter[1],
                                                 wordsafter[0] + ' ' + wordsafter[1]+' '+wordsafter[2]],
                                                countNextWords)
            if(len(wordsafter)>1 and len(wordsafter)<3):
                #check 3 words first, second, first&second
                countNextWords=findNextCountsForWords([wordsafter[0],wordsafter[1],wordsafter[0]+' '+wordsafter[1]],countNextWords)
            if(len(wordsafter)==1):
                countNextWords=findNextCountsForWords([wordsafter[0]],countNextWords)

        # if(len(words[0])>1):
        #     wordsbefore=words[0].split()
        #     if (len(wordsbefore) >1):
        #         # check 3 words first, second, first&second
        #         countPrevWords = findPrevCounts([wordsbefore[0], wordsbefore[1], wordsbefore[0] + ' ' + wordsbefore[1]],
        #                                          countPrevWords, initalVal)
        #     if (len(wordsbefore) == 1):
        #         countPrevWords = findPrevCounts([wordsbefore[0]], countPrevWords, initalVal)
        #     print(wordsbefore)
    outputNextWords=sorted(countNextWords.items(), key=lambda x: x[1], reverse=True)
    # outputPrevWords=sorted(countPrevWords.items(),key=lambda x: x[1], reverse=True)

    return(outputNextWords)


getPossibleNextWordsAfterPrticularWord(result,"check")



#check what is the original look like
# for rr in result:
#     originalTxt, cleanedUpTxt = getCleanedUpTextString(rr[0])
#     saveOriginal=originalTxt
#     originalTxt = re.sub('[a-zA-Z0-9]+-[0-9A-Za-z]+-[0-9a-zA-Z]+', '', originalTxt)
#     #
#     # # remove every thing other than alphabets
#     originalTxt = re.sub("[^a-zA-Z]", ' ', originalTxt)
#     # remove extra two spaces
#     originalTxt = originalTxt.replace('  ', ' ')
#     # convert the asset names
#     originalTxt = getReplacedWithAssets(originalTxt)
#     print(saveOriginal,"....",originalTxt)


finaloutput={}
for tt in sortedCountNGrams[0:250]:
    inputString=tt[0]
    output=getPossibleNextWords(result,inputString,tt[1])
    # remove the tuples which have percentage less than 0.6
    filteredOutput=[]
    for kk in output[0:4]:
        if (kk[1] > 0.4):
            filteredOutput.append(kk)

    # tokenisedMatchString = inputString.split()
    # if ((tokenisedMatchString[0] == 'check') and (tokenisedMatchString[1] == 'record')):
    #     inputString =

    inputString=inputString.replace("check record","check")
    print(inputString,"(",tt[1],")=>",json.dumps(filteredOutput))
    finaloutput[inputString]={"original":tt,"final":json.dumps(output[0:4])}

print(finaloutput)


###
#check if bigram has a preopsition and a word then most likely you should consider it
#check if the percentage is <60 you should ignore
#
###


import nltk
text=nltk.word_tokenize("damage")
print(nltk.pos_tag(text))


getPossibleNextWords(result,"clean work area")
print(finaloutput)
txt="wordsafter are instepecjj"
gg=txt.split()
gg[0]
#####################################################################################
'''

Some observations as on 13-march:
If I am using the Ngrams created using all the cases . and then apply them all the Pump cases fro the trienergy versus I do 10000
random cases the frequency disrtribution is almost same which signifies that I am taking a good sample and and the sample is
representive of the actual distributions

This distribution is different from if you do DIstinct().

The word disrtibution is different when I have a smaller list of the Ngram (basically formed while taking only Dictinct WO)

So we have 4 scenarios
1. all WO + large Ngram
2. sample WO + large Ngram
3. distinct Wo + large Ngram
4. All WO + small Ngram
5. Distinct WO + small Ngram


1 & 2 matches
4 & 5 are different




'''




''''EXTRA NOT TO BE CONSIDERED'''''
###############################
#result[1][0]

for rr in result[1:100]:
    #print(rr[0])

    originalTxt, cleanedUpTxt = getCleanedUpTextString(rr[0])
    print(originalTxt,"...",cleanedUpTxt)
    cleanedUpTxt=cleanedUpTxt.replace('  ','')
    uniArr = getUnigramDistributionFromText(cleanedUpTxt)
    print(uniArr)

txt=result[9][0]
getReplacedWithAssets(txt.lower())

for aa in assets:
    st='"'+assets[aa]+'",4'
    print(str)
    #txt = txt.replace(aa[0], aa[1])





'''
I shoudld create the bigrams on the basis of the prob of occurence of the next word
if the prob is very high that the two words come always together then even later when we are finaliing the results
we can replace the unigram to bigrams
e.g drain is drain point most common than drain



t seems the maintainable item are the various assets that are being talked about
so we need to see what all words have been picked in the Ngrams and do they qualify as item

I guess our bigrams should be properly created to get better resulsts

we just need to find out those guys.

'''