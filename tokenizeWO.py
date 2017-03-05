'''
This code will find out the Ngrams for all the cases and store those Ngrams as a statement in the database
These Ngram statements will then be pulled from the DB and then analyzed for the chi-square association.
'''
from nltk.util import bigrams
import re
import json
import mysql.connector


'''
Fetch the WO from the DB
'''
def getWO(cnx):
    result = []
    try:
        cursor = cnx.cursor()
        cursor.execute("""SELECT WH_REQUEST_SITE_PK,WH_ORIG_RQST_DESC_C from workorders_assets """)
        result = cursor.fetchall()
       # print(result)
    finally:
        cnx.close()
    return(result)

'''
Read the uigram & bigrams dictionaries
'''
def getIntiialize(conn):
    results=getWO(conn)
    lstWorkOrders=[]
    for res in results:
        lstWorkOrders.append({"ID":res[0],"WO":res[1]})



    unigramFile = "/Users/305015992/pythonProjects/assetAnswer/unigramWO.csv"
    bigramFile= "/Users/305015992/pythonProjects/assetAnswer/bigrams.csv"
    lstUnigrams=[]
    lstBigrams=[]
    #trigramFile= "/Users/305015992/pythonProjects/wordcloud/trigramsWO.csv"
    for ll in open(unigramFile):
        llArr = ll.split(",")
        key = llArr[0].replace('"', '')
        lstUnigrams.append(key)

    for ll in open(bigramFile):
        llArr = ll.split(",")
        key = llArr[0].replace('"', '')
        lstBigrams.append(key)

    return (lstWorkOrders,lstUnigrams,lstBigrams)

'''
given the WO find the unigrams
'''
def findAllUnigramsOfWO(WO,lstUnigrams):
    #print(WO)
    arrUnigramWords = WO.split(" ")
    lstFinal=[]
    for uni in arrUnigramWords:
        uni = uni.lstrip()
        uni = uni.rstrip()
        if uni in lstUnigrams:
            # print(bi,dictBigrams[bi])
            lstFinal.append(uni)

    return(', '.join(lstFinal))

'''
given the WO find the bigrams
'''
def findAllBigramsofWO(WO,lstBigrams):
    bigrm = list(bigrams(WO.split()))
    # print(bigrm)
    arrBigramWords = ', '.join(' '.join((a, b)) for a, b in bigrm)

    lstFinal=[]
    for bi in arrBigramWords.split(','):
        bi = bi.lstrip()
        bi = bi.rstrip()
        if bi in lstBigrams:
            # print(bi,dictBigrams[bi])
            lstFinal.append(bi.replace(' ','_'))

    return (', '.join(lstFinal))

'''
for a WO find both the unograms and bigrams and store them as a normalized version of the WO
'''
def getNormalizedCases(cnx):
    lstWorkOrders, lstUnigrams, lstBigrams=getIntiialize(cnx)
    arrUnigramFiltered = {}
    arrFinalOutput=[]
    count=0
    for woObj in lstWorkOrders:
        print(count)
        count=count+1
        #woObj=lstWorkOrders[1]
        # print("before {}",case)
        #key=13157
        case = woObj['WO']
        #print(case)
        case=case.lower()
        #print(case)
        case = case.strip();
        case = re.sub('/[^A-Za-z0-9 _\-\+\&\,\#]/', '', case)
        case = case.replace('"', ' ')
        case = case.replace('\"', ' ')
        case = case.replace('>', ' ')
        case = case.replace('@', ' ')
        case = case.replace('<', ' ')
        case = case.replace(':', ' ')
        case = case.replace('.', ' ')
        case = case.replace('(', ' ')
        case = case.replace(')', ' ')
        case = case.replace('[', ' ')
        case = case.replace(']', ' ')
        case = case.replace('_', ' ')
        case = case.replace(',', ' ')
        case = case.replace('#', ' ')
        case = case.replace('-', ' ')
        case = case.replace('/', ' ')
        case = case.replace('"', ' ')
        case = case.replace('\n', ' ')
        case = case.replace('~', ' ')
        case = case.replace('\r', ' ')
        case = case.replace('%', ' ')
        case = case.replace('$', ' ')
        case = case.replace('!', ' ')
        case = case.replace('*', ' ')

        case = re.sub(r'\d+', ' ', case)
        #print("case",case)
        #for our purpose we will just list down the possible unigramsn and bigrams of the sentece and
        #keep the sentence as the sum of both the grmas
        #arrTempTerms = case.split(" ")
        unigramWords=findAllUnigramsOfWO(case,lstUnigrams)
        #print("Uni",unigramWords)
        bigramWords=findAllBigramsofWO(case,lstBigrams)
        #print("Bigram",bigramWords)
        finalWords=unigramWords+","+bigramWords
        #print("finalWords",finalWords)
        arrFinalOutput.append({"ID": woObj['ID'], "WO": woObj['WO'],"ngrams":finalWords})

    #print(arrFinalOutput)
    return (arrFinalOutput)



'''Main code '''
#this is how you can connect to the mysql database
cnx = mysql.connector.connect(user='root', password='root',host='localhost',database='assetanswers')
#lstWorkOrders, lstUnigrams, lstBigrams=getIntiialize(cnx)
#print(lstUnigrams)
#findAllUnigramsOfWO("this pump fail start",lstUnigrams)
output=getNormalizedCases(cnx)

print(output[1:1000])


'''storing the results to the db'''
conn = mysql.connector.connect(user='root', password='root',host='localhost',port='3306',database='assetanswers')
cur = conn.cursor()
for woObj in output:
    cur.execute("""
       UPDATE workorders_assets
       SET WH_NGRAMS=%s  WHERE WH_REQUEST_SITE_PK=%s
    """, (woObj['ngrams'],woObj['ID']))

    conn.commit()





'''''EXTRA'''

'''It seems I am not able to run the update query ..so I will for now save the information to  file an then use PHP code to perform the update'''
fp = open("/Users/305015992/pythonProjects/assetAnswer/ngramsWO.txt", "w")
for woObj in output:
    str=woObj['ID']+"~ROHIT~"+woObj['ngrams']
    fp.write(str)
    fp.write("\n")
fp.close()



#try to comeup with the chi square thing
from sklearn.feature_extraction.text import CountVectorizer
lines = []
for wobj in output:
    lines.append(' '.join(wobj['ngrams'].split(',')))  # now we need to vectorize the corpus

print(lines[1:100])
count_vect = CountVectorizer(min_df=0.006, strip_accents='unicode', binary=False)
rawdtm = count_vect.fit_transform(lines)
vocab = count_vect.get_feature_names()
# convert the dtm to regular array
countDtm = rawdtm.toarray()
# convert the dtm to numpy array
import numpy as np
countDtm = np.array(countDtm)
print(countDtm)





# print(output[1:1000])

# print(lstBigrams)
# txt="change oil in the bearing"
# bigrm = list(bigrams(txt.split()))
# arrBigramWords = ', '.join(' '.join((a, b)) for a, b in bigrm)
# print(arrBigramWords)
#
# lstFinal=[]
# for bi in arrBigramWords.split(','):
#     print(bi)
#     bi = bi.lstrip()
#     bi = bi.rstrip()
#     if bi in lstBigrams:
#         # print(bi,dictBigrams[bi])
#         lstFinal.append(bi)
#
# print(lstBigrams)




