'''
This script is to find out the issues disrtibution for a particular site and equipment.
It takes as an input the unigram list, bigram list and trigram list
It makes use of clustering since all the wor are considered even if they are duplicate
'''
import numpy as np
from nltk.util import bigrams
from nltk.util import trigrams
import csv
from nltk.corpus import stopwords
from sklearn.cluster import KMeans
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
unigramFile=prefixpath+"unigramWO.csv"
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
undesiredWordsFile=prefixpath+"UndesiredWords.csv"
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

        #val=llArr[1].replace('"','')

print(arrUndesiredWords)

'''
read the bigrams with their frequecny
'''
bigramFile=prefixpath+"bigrams.csv"
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
trigramFile=prefixpath+"trigramsWO.csv"
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
Use the database to fetch the Workorders that you want to analyze. Currently they are based on the equipment type
'''
def getData(equipmentType,siteName):

    cnx = mysql.connector.connect(user='root', password='root',host='localhost',port='3306',database='assetanswers')

    result=[]
    try:
       cursor = cnx.cursor()
       #No Distinct all the workorders are read
       sqlQuery="SELECT WH_ORIG_RQST_DESC_C from workorders_assets where EQ_EQ_CLASS_C='%s' and WH_SITE_C='%s'" \
                %(equipmentType,siteName)
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

'''
getCleanedUpTextString: You will perform tyhe basic cleanup ...numbers, punctuations, stopwords...Note that the stopwords should be quite
similar to what were used at the time of creating the dictionary otherwise some of the bigrams will not match
'''
def getCleanedUpTextString(txt):
    txt = txt.lower()
    original = txt
    #first remove the code FWA-P-201A in the strings and may be we can store the stuff as well
    txt = re.sub('[a-zA-Z0-9]+-[0-9A-Za-z]+-[0-9a-zA-Z]+', '', txt)

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
getTotalDistribution: Get the final count of the various Ngrams in the corpus. So what is the final frequency for the bigram words,, trigram words
'''
def getTotalDistribution(ngramArray,finalCountDict):
    for nn in ngramArray:
        if nn[0] in countNgrams:
            finalCountDict[nn[0]] += float(1)
        else:
            finalCountDict[nn[0]] = float(1)

    return(finalCountDict)


'''
drawFreqDistribution: Draws the frequecny distribution in the form of a bar graph
'''
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

    for rect in winners:
        # print(rect)
        width_rect = int(rect.get_width())
        # print(width_rect)
        hcap = str(width_rect)
        ax.text(width_rect + 2, rect.get_y() - 0.25, hcap, ha='center', va='bottom', rotation="horizontal")

    plt.show()
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


'''CLUSTERING STUFF'''

'''
getVectorized:Convert the text to tfidf matrix
'''
def getVectorized(normalizedWorkOrders):

    lines = []
    countToCaseIdMap = {}
    # # maximum is 4997
    count = 0
    for wo in normalizedWorkOrders:
        wo=wo.replace(',',' ')
        lines.append(wo)  # now we need to vectorize the corpus
        #countToCaseIdMap[count] = key
        #count = count + 1


    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = TfidfVectorizer(min_df=0.006, stop_words=stops, strip_accents='unicode', norm='l2',
                                 sublinear_tf=True)


    tfSparseMatrix = vectorizer.fit_transform(lines)
    tfDtm = tfSparseMatrix.toarray()
    tfDtm = np.array(tfDtm)


    from sklearn.feature_extraction.text import CountVectorizer
    count_vect = CountVectorizer(min_df=0.006, stop_words=stops, strip_accents='unicode', binary=False)
    rawdtm = count_vect.fit_transform(lines)
    vocab = count_vect.get_feature_names()
    # convert the dtm to regular array
    countDtm = rawdtm.toarray()
    # convert the dtm to numpy array
    countDtm = np.array(countDtm)
    #print(countDtm)
    # need to convert it to numpy array so that we can easily perform the operations on it
    vocab = np.array(vocab)
    return(tfSparseMatrix,count_vect,countDtm)

'''
performKmeans:Perform kmeans clustering with cluster=3
'''
def performKmeans(tfSparseMatrix):

    # the max_iter is how many iterations before the convergence is assumed
    # n_init is the number of times the algo is run
    K_Cluster = 3
    km = KMeans(n_clusters=K_Cluster, init='k-means++', max_iter=1000, n_init=10, verbose=False)
    # you need to call the km.fit_predict so that the kmeans cane be run and then each of the points can be assigned a cluster index
    km.fit_predict(tfSparseMatrix)
    return(km)

'''
getCasesDistributionPerCluster: Find out how the cases are distributed in the clusters
'''
def getCasesDistributionPerCluster(km):
    casesPerCluster = {}
    for i, cluster in enumerate(km.labels_):
        if cluster not in casesPerCluster.keys():
            casesPerCluster[cluster] = str(i)
        else:
            casesPerCluster[cluster] += "," + str(i)
    return(casesPerCluster)


'''
findMostCommonWords:This code is suppose to find the top words in the cases
The cases already contains the unigrams and ngrams. All we need to do is to count them
'''
def findMostCommonWords(normalizedWO):
    countNgrams={}
    for wo in normalizedWO:
        ngrams = wo.split(',')
        for nn in ngrams:
            nn = nn.strip()
            if (len(nn) > 1):
                if nn in countNgrams:
                    countNgrams[nn] += float(1)
                else:
                    countNgrams[nn] = float(1)
    sortedCountNGrams = sorted(countNgrams.items(), key=lambda x: x[1], reverse=True)
    return(sortedCountNGrams)



'''
Main function:
Read the workorders from the database
find out the unigrams, bigrams,trigrams
and store them into a file for further analysis

'''

import json
fp = open(prefixpath+"cognitiveOutput.txt", "w")
outputResult=[]
result=getData("Pump","BangBo")
#print(result)
#count the number of times each word has come...this way we can give a distribution and may be that is helpful
countNgrams={}
normalizedWorkOrders=[]
for rr in result:
    originalTxt, cleanedUpTxt=getCleanedUpTextString(rr[0])

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
        if(uu[0]=="change"):
            print("change",originalTxt)
            print("\n")

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



    stringifiedTrigram = ', '.join(tt[0].replace(' ','_') for tt in triArr)
    stringifiedBigram=', '.join(bb[0].replace(' ','_') for bb in finalbb)
    stringifiedUnigram=', '.join(uu for uu in finUnigrams)

    normalizedWO=stringifiedUnigram+','+stringifiedBigram+','+stringifiedTrigram
    if(len(normalizedWO.strip())>2):
        normalizedWorkOrders.append(normalizedWO)
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
#print(sortedCountNGrams[0:100])
toplabels=drawFreqDistribution(sortedCountNGrams)
fp.write(json.dumps(outputResult))
fp.write("..........\n")
fp.write(json.dumps(toplabels))
fp.close()


###now go for the clustering
tfSparseMatrix, count_vect, countDtm = getVectorized(normalizedWorkOrders)
km=performKmeans(tfSparseMatrix)

clusterDistribution = np.unique(km.labels_, return_counts=True)
numberOfClusters=len(clusterDistribution[0])
print(clusterDistribution)
casesDistributionPerCluster=getCasesDistributionPerCluster(km)
print(casesDistributionPerCluster)

finalFrequencyWords={}
for clustNum in casesDistributionPerCluster:
    indexes=casesDistributionPerCluster[clustNum]
    print(indexes)
    normalizedWoSubset = [normalizedWorkOrders[int(i)] for i in indexes.split(',')]
    orderedFreq=findMostCommonWords(normalizedWoSubset)[1:50]
    print(orderedFreq)
    finalFrequencyWords[clustNum]=orderedFreq


#based on the proportion of the cluster we will pick the words
tt=(clusterDistribution[1]/np.sum(clusterDistribution[1]))*50
#convert to integer
tt=tt.astype(int)

prepareDataForPlotting=[]
for key in finalFrequencyWords:
    totWords=tt[key]
    #for i in totWords:
    #   prepareDataForPlotting.append(finalFrequencyWords[0:])
    prepareDataForPlotting.append(finalFrequencyWords[key][0:totWords])

print(prepareDataForPlotting)

flatten = lambda l: [item for sublist in l for item in sublist]


prepareDataForPlotting=flatten(prepareDataForPlotting)
drawFreqDistribution(prepareDataForPlotting)