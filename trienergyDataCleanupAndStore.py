###
#This script  contains bunch of sections which were used to perform the database insertions for WO
# The sections are finding the names of the assets, adding the WO data to the database for pump repair events and for
# all the events reported
###


import numpy as np
from nltk.util import bigrams
from nltk.util import trigrams
import csv
from nltk.corpus import stopwords
import re
import mysql.connector

'''
Read the short description that were already copied to the file manuall and clean them up to get the name of the asset
'''
prefixpath="/Users/305015992/pythonProjects/assetAnswer/"
fname=prefixpath+"WO_short_desc_trienergy.txt"
with open(fname, 'r') as myfile:
    data=myfile.read()
#len(data)

data.split("\n")
stopwords=['a','b','c','d','e','f','g']

assetNames=[]
for dd in data.split('\n'):
    #print(dd)
    dd=re.sub('[a-zA-Z0-9]+-[0-9A-Za-z]+-[0-9a-zA-Z]+', '', dd)
    dd = re.sub("[^a-zA-Z]", " ", dd)
    dd = dd.strip()
    arrtxt = dd.split()
    # remove stop words
    # stops = set(stopwords.words("english"))
    words = [w for w in arrtxt if not w in stopwords]
    words = [w for w in words if len(w) > 1]
    finalString = ' '.join(words)

    #print(finalString)
    assetNames.append(finalString)


'''
This will print the asset names that you can copy into some file
'''
uniqueAssets=set(assetNames)
for hh in uniqueAssets:
    print(hh)



#########################################################

'''
This section will read the pump repair events that are stored in the text file WO_DESC_Trienergy_PUMP_REPAIR_EVENTS
It does the basic cleaning job and then stores it in the database table
'''

fname=prefixpath+"WO_DESC_Trienergy_PUMP_REPAIR_EVENTS.txt"
fp = open(prefixpath+"insertQueries.txt", "w")

with open(fname, 'r') as myfile:
    data=myfile.read()


cnx = mysql.connector.connect(user='root', password='root', host='localhost', port='3306', database='assetanswers')

for dd in data.split("\n"):
    original=dd
    original=original.replace('\'','')
    original=original.replace(',',' ')
    dd = re.sub('[a-zA-Z0-9]+-[0-9A-Za-z]+-[0-9a-zA-Z]+', '', dd)
    dd = re.sub("[^a-zA-Z]", " ", dd)
    dd = dd.strip()
    #arrtxt = dd.split()
    # remove stop words
    # stops = set(stopwords.words("english"))
    #finalString = ' '.join(words)
    cursor = cnx.cursor()
    sqlQuery = "INSERT INTO `TrienergyWO`(`WO_Long_Desc`, `WO_Normalized_Desc`) VALUES ('%s','%s')" \
                   % (original,dd)
    fp.write(sqlQuery)
    fp.write(";\n")
    cursor.execute(sqlQuery)
    cnx.commit()
cnx.close()


#############################################################
'''
This section of the code is to store all the trienergy data to the database. Th ebasic normalization that is
detection of the asset names to in the text has been done. This will be used when we are doing the Ngram analysis to get
unigrams, bigrams, trigrams

'''

fname=prefixpath+"trienergyNgram/assetNames.txt"
with open(fname, 'r') as myfile:
    data=myfile.read()

#read the asset names
assets={}
for dd in data.split("\n"):
    dd=dd.lower()
    assets[dd]=dd.replace(' ','_')

assets


def getReplacedWithAssets(txt):
    for aa in assets:
        txt=txt.replace(aa,assets[aa])
    return(txt)



fname=prefixpath+"WO_DESC_Trienergy_forNgram.txt"
with open(fname, 'r') as myfile:
    data=myfile.read()
cnx = mysql.connector.connect(user='root', password='root', host='localhost', port='3306', database='assetanswers')
count=0
for dd in data.split("\n"):
    count=count+1
    dd=dd.lower()
    original=dd
    original=original.replace('\'','')
    original=original.replace(',',' ')
    dd = re.sub('[a-zA-Z0-9]+-[0-9A-Za-z]+-[0-9a-zA-Z]+', '', dd)
    dd = re.sub("[^a-zA-Z]", " ", dd)
    dd=dd.replace('  ',' ')
    dd = dd.strip()

    dd = getReplacedWithAssets(dd)
    print(original,"...",dd)
    #arrtxt = dd.split()
    # remove stop words
    # stops = set(stopwords.words("english"))
    #finalString = ' '.join(words)
    cursor = cnx.cursor()
    sqlQuery = "INSERT INTO `TrienergyWOforNgrams`(`WO_Desc`,`WO_Norm_Desc`) VALUES ('%s','%s')" \
                   % (original,dd)
    #fp.write(sqlQuery)
    # #fp.write(";\n")
    cursor.execute(sqlQuery)
    cnx.commit()

    # if(count>100):
    #     break;

cnx.close()



########
'''
This section is to read the unigrams found earlier using the createNgrams script and save them to the database. Actually
we could do that in that script as well
'''

fname=prefixpath+"trienergyNgram/unigrams_trienergy_R.txt"
with open(fname, 'r') as myfile:
    data=myfile.read()


print(data)
cnx = mysql.connector.connect(user='root', password='root', host='localhost', port='3306', database='assetanswers')
count=0
for dd in data.split("\n"):
    ddd=dd.split(',')
    unigram=ddd[0]
    unigram=unigram.replace('\'','')
    print(unigram)
    cursor = cnx.cursor()
    sqlQuery = "INSERT INTO `Ngrams`(`ngram`) VALUES ('%s')" \
                   % (unigram)
    cursor.execute(sqlQuery)
    cnx.commit()

cnx.close()

