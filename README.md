Python Implementation of a ranking model combining document content-based 
similarity and semantic descriptions (nGBR).

nGBR maps a collection of documents to a graph, where each state is a document, edges 
are weighted by content-based similarity between documents, and states are represented 
by semantic tags. The structure is similar to an hidden Markov model (HMM).

Given a query in the form of either tags (semantic retrieval) or documents (query-by-example), 
the ranking algorithm (similar to the Viterbi algorithm for HMM) walks through the model 
and finds the best path combining both content and contextual information.

More details of this work are available in:

Miotto, R. and Orio, N. (2012)
A Probabilistic Model to Combine Tags and Acoustic Similarity for Music Retrieval
ACM Transactions of Information Systems,
Vol. 30(2):8

Please cite this paper if you use the code.

Python Requirements: numpy

Author: Riccardo Miotto
