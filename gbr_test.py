'''
 Experimental framework using GraphBasedRanking to rank a list of documents

 @Note-1: extend the code to load real data (real similarity matrix and semantic multinomials)

 @Note-2: no evaluation infrastructure

 @data n: number of songs in the collection
 @data cbsim: "n * n" matrix with pairwise audio content-based similarity between documents (absolute value) to weight the edges of the graph
 @data smn: "n * m" matrix with semantic description of each song in the model (normalized to sum-one)

'''

from gbranking import GraphBasedRanking
import numpy as np


# parameters
nsong = 500
ntag = 50

# random content-based similarity matrix
cbsim = np.random.random ([nsong, nsong])

# random semantic multinomials
smn = np.random.random ([nsong, ntag])
smn /= smn.sum(axis=1)[:,None]

# init model
gbr = GraphBasedRanking (cbsim, smn)

# semantic retrieval from tag (5)
print 'playslist generated from tag (5)'
ranklist = gbr.semantic_retrieval ([5], 100)
print ranklist
print ''

# semantic retrieval from tags (0, 8, 45, 27)
print 'playslist starting from tags (0, 8, 45, 27)'
ranklist = gbr.semantic_retrieval ([0, 8, 45, 27], 100)
print ranklist
print ''

# query-by-example from song (0)
print 'playslist starting at song (0)'
ranklist = gbr.qbe_retrieval (0, 100)
print ranklist
print ''

# query-by-example from song (45)
print 'playslist starting at song (45)'
ranklist = gbr.qbe_retrieval (45, 100)
print ranklist
print ''
