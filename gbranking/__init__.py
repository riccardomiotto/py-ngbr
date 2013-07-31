'''
 Graph-based Ranking Model

 @input a: N * N matrix with content-based similarity between documents
 @input b: N * M matrix with the semantic representation of the N documents over M tags (semantic multinomial, normalized to sum-1)

 @output: ranking list of documents combining content-based similarity and tags
 
 @author: Riccardo Miotto
'''

from numpy import array, ndarray, log, argmax, ceil, floor, argsort, zeros, exp


class GraphBasedRanking (object):

    def __init__ (self, a, b):
        self.a = a
        self.b = b / b.sum(axis=1)[:,None]
        self.n, self.m = self.b.shape
        self.a[range(self.n), range(self.n)] = 10e-10



    '''
    semantic retrieval (query-by-tag)

    @param q: list of query tags expressed as indices in the SMNs (i.e., [0, ..., M-1])
    @param lres: number of documents to retrieve
    '''
    def semantic_retrieval (self, q, lres = 100):
        # filter transitions
        ntrans = ceil (self.n * 0.4)
        afilt = self.__apreproc (ntrans)

        # observations
        obs = log(self.b[:,q].prod(axis=1))
    
        # get initial state
        pi = array ([1/float(self.n) for i in xrange(self.n)])

        # return ranklist 
        return self.__ranking (afilt, obs, pi, lres)


    
    '''
    query-by-example (the query is a document in the model)

    @param q: query expressed as state number (index in the transition matrix, [0,...,N-1])
    '''
    def qbe_retrieval (self, q, lres = 100):
        # filter transitions
        ntrans = ceil (self.n * 0.8)
        afilt = self.__apreproc (ntrans)

        # get initial state
        pi = array ([10e-10 for i in xrange(self.n)])
        pi[q] = 1

        # get observations
        obs = array([log(self.__smn_similarity(self.b[i],self.b[q])) for i in xrange(self.n)])

        # return ranklist
        return self.__ranking (afilt, obs, pi, lres)


        
    '''
     private functions
    '''
    
    # pre-process edge matrix by retaining only the "nth" most similar documents
    def __apreproc (self, nth):
        sind = argsort (self.a, axis = 0)[::-1]
        afilt = self.a
        for i in xrange(afilt.shape[1]):
            afilt[sind[nth:,i], i] = 10e-10
            afilt[:,i] /= afilt[:,i].sum()
        return afilt

    
    '''
    walk thorugh the model and rank the documents

    @algorithm: split retrieval in several sub-retrieval
    '''
    def __ranking (self, a, obs, pi, lres):
        tstep = 50
        sublist = 25

        # iteration
        ranklist = []
        while len(ranklist) < lres:
            # update model
            if len(ranklist) > 0:
                # re-start walk from last state
                pi[:] = 10e-10
                pi[ranklist[-1]] = 1
                # remove selected states from the reachable set
                a[ranklist,:] = 10e-10
                a /= a.sum(axis=0)

            # forward computation
            atmp = a
            delta = zeros (self.n)
            deltat = log(pi) + obs
            psi = zeros ((self.n, tstep), int)
            for t in xrange(1,tstep):
                for i in xrange(self.n):
                    psi[i,t] = argmax (deltat + log(atmp[:,i]))
                    delta[i] = deltat[psi[i,t]] + log(atmp[psi[i,t],i]) + obs[i]
                    atmp[psi[i,t],i] /= 10
                deltat = delta
                atmp /= atmp.sum(axis=0)

            # backward computation
            q = zeros((tstep,1), int)
            q[-1] = argmax (deltat)
            for t in xrange(1,tstep):
                q[tstep-1-t] = psi[q[tstep-t], tstep-t]

            # select documents
            ndoc = 0
            for i in xrange(len(q)):
                if ndoc == sublist:
                    break
                if q[i] in ranklist:
                    continue
                ranklist.append (int(q[i]))
                ndoc += 1
            
        # return the desired number of songs
        if len(ranklist) > lres:
            ranklist = ranklist[:lres]
        return ranklist


    # similarity between SMNs
    def __smn_similarity (self, smn1, smn2):
        kldiv = 0
        for i in xrange(smn1.shape[0]):
            kldiv += smn1[i] * (log(smn1[i]) - log(smn2[i]))
        return exp(-kldiv)
    
