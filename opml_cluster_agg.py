# -*- mode: Python; tab-width: 2; indent-tabs-mode:nil; -*-          #
######################################################################
# Author: Anton Strilchuk <ype@env.sh>                               #
# URL: http://isoty.pe                                               #
# Created: 07-04-2014                                                #
# Last-Updated: 30-07-2014                                           #
#   By: Anton Strilchuk <ype@env.sh>                                 #
#                                                                    #
# Filename: opml_cluster_agg                                         #
# Version: 0.0.1                                                     #
# Description: Clustering RSS Aggregator                             #
# Based On: Carl Anderson's wonderful Howto @ http://bit.ly/1ioH5pY  #
######################################################################
######################################################################
from xtermcolor import colorize

##,----
##| OPML/Feed Links
##| Set Path to OPML File, or manual
##| enter feeds in format:
##| feeds = ['http://path/to/feed',
##|          'http://path/to/next/feed']
##| NOTE: large opml files will take
##|       longer to parse and cluster
##`----
import listparser
# opml = listparser.parse('./agg_links.opml')
# feeds = []
# for i in opml.feeds:
#     feeds.append(i.url.encode('utf-8'))

feeds = ['http://feeds.feedburner.com/uk/gizmodo?format=xml',
         'http://www.engadget.com/tag/rss/?format=xml',
         'http://feeds.feedburner.com/T3/news?format=xml',
         'http://feeds.theguardian.com/theguardian/technology/rss',
         'http://feeds.gawker.com/lifehacker/full?format=xml',
         'https://news.ycombinator.com/rss?format=xml',
         'http://laughingsquid.com/feed/']

##,----
##| PARSE
##| parses feeds to set
##| of words per document (post)
##`----
import feedparser
import nltk
corpus = []
titles=[]
hyperlinks=[]
ct = -1
for feed in feeds:
    d = feedparser.parse(feed)
    for e in d['entries']:
        #working on function to constrain dates to specified time range
        #if e['published'] <= "Fri, 04 Apr 2014":
            words = nltk.wordpunct_tokenize(nltk.clean_html(e['description']))
            words.extend(nltk.wordpunct_tokenize(e['title']))
            lowerwords=[x.lower() for x in words if len(x) > 1]
            ct += 1
            print (colorize(ct, ansi=5)),'\t',
            print (colorize(e['title'].encode("utf-8"), ansi=9))
            print ('\t'),
            print (colorize(e['link'].encode("utf-8"), ansi=4))
            corpus.append(lowerwords)
            titles.append(e['title'])
            hyperlinks.append(e['link'])

##,----
##|  tf-idf implementation
##|  from http://timtrueman.com/a-quick-foray-into-linear-algebra-and-python-tf-idf/
##`----
import math
from operator import itemgetter
def freq(word, document): return document.count(word)
def wordCount(document): return len(document)
def numDocsContaining(word,documentList):
    count = 0
    for document in documentList:
        if freq(word,document) > 0:
            count += 1
    return count
def tf(word, document): return (freq(word,document) / float(wordCount(document)))
def idf(word, documentList): return math.log(len(documentList) / numDocsContaining(word,documentList))
def tfidf(word, document, documentList): return (tf(word,document) * idf(word,documentList))

##,----
##| KEYWORDS EXTRACTION
##| extracts the top keywords from each doc
##| This defines features of a common feature vector
##`----
import operator
def top_keywords(n,doc,corpus):
    d = {}
    for word in set(doc):
        d[word] = tfidf(word,doc,corpus)
    sorted_d = sorted(d.iteritems(), key=operator.itemgetter(1))
    sorted_d.reverse()
    return [w[0] for w in sorted_d[:n]]

key_word_list=set()
nkeywords=2
[[key_word_list.add(x) for x in top_keywords(nkeywords,doc,corpus)] for doc in corpus]

ct=-1
for doc in corpus:
    ct+=1
    print (colorize(ct, ansi=5)), '\t',
    print (colorize("KEYWORDS", ansi=9))
    print ('\t'),
    print (colorize(" ".join(top_keywords(nkeywords,doc,corpus)).encode('utf-8'), ansi=4))

##,----
##| VECTOR CONVERSION
##| Turn each doc into a feature vector
##| using TF-IDF score
##`----
feature_vectors=[]
n=len(corpus)

for document in corpus:
    vec=[]
    [vec.append(tfidf(word, document, corpus) if word in document else 0) for word in key_word_list]
    feature_vectors.append(vec)

##,----
##|  symmatrix matrix of
##|  cosine similarities
##`----
import numpy
from nltk import cluster
mat = numpy.empty((n, n))
for i in xrange(0,n):
    for j in xrange(0,n):
        mat[i][j] = nltk.cluster.util.cosine_distance(feature_vectors[i],feature_vectors[j])

##,----
##|  Hierarchically Cluster mat
##`----
from hcluster import linkage
t = 0.9
Z = linkage(mat, 'single')
#dendrogram(Z, color_threshold=t)

#import pylab
#pylab.savefig( "new_agg_cluster.png" ,dpi=800)

##,----
##|  Cluster Extraction
##`----
def extract_clusters(Z,threshold,n):
    clusters={}
    ct=n
    for row in Z:
        if row[2] < threshold:
            n1=int(row[0])
            n2=int(row[1])

            if n1 >= n:
                l1=clusters[n1]
                del(clusters[n1])
            else:
                l1= [n1]

            if n2 >= n:
                l2=clusters[n2]
                del(clusters[n2])
            else:
                l2= [n2]
            l1.extend(l2)
            clusters[ct] = l1
            ct += 1
        else:
            return clusters

clusters = extract_clusters(Z,t,n)

for key in clusters:
    print (colorize("|-------| ============================================", ansi=10))
    for id in clusters[key]:
        print (colorize ('|', ansi=11)),
        print (colorize(id, ansi=5)), '\t',
        print (colorize ('|', ansi=11)),
        print (colorize(titles[id].encode('utf-8'), ansi=9))
        print (colorize ('|-------|', ansi=11)),
        print (colorize(hyperlinks[id], ansi=4))
