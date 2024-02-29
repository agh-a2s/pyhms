# =============================================================================
# Calculates the performance indicator robust peak ratio the results for static 
# or dynamic optimization. The static method calc_RPR(X,foundEval,tolFunRPR,problem) 
# from this class calculates the performance indicator and the difference between
# the value of the reported solutions and the actual global minimum value
# =============================================================================

class PerformIndicator:    
    @staticmethod
    def calc_RPR(X,foundEval,tolFunRPR,problem):
        ''' calculates robust peak ratio for the static and dynamic problem '''
        # Output:
            # RPR: Robust peak ratio given the loosest and tightest function tolerances.
                # if the problem is static, it is a scalar. It it is dynamic, it is a 1-D array, 
                # indicating the performance for each time step 
            # valDiff: the difference between the global minimum value and the value of 
                # if the problem is dynamic, it is an array showing the indicator for each time step and each global minimum
        # The inputs are:
            # X (2D array): solutions reported as global minima during the optimization process
            # foundEval (1D array): function evaluation at which these solutions were found
            # tolFunRPR (1D array): the loosest  and tightest tolerance for optimality
            # problem (object): optimization problem  
            
        if problem.isDynamic: # dynamic problem?
            # preallocate 
            RPR=np.zeros(problem.dynaAttr.numTimeStep)
            valDiff=np.inf*np.ones((problem.dynaAttr.numTimeStep,problem.statData.numGlobMin))
            # calculate the RPR and valDiff for each time step
            for timeStep in np.arange(problem.dynaAttr.numTimeStep):
                # find the start and end function evaluation for each time step
                if timeStep==0:
                    startFE=0
                    finishFE=problem.statData.maxEval
                else:
                    startFE= problem.statData.maxEval+(timeStep-1)*problem.dynaData.chFr
                    finishFE= startFE+problem.dynaData.chFr
                # select the solutions that have been found in this time step
                relInd=np.logical_and(startFE<foundEval,finishFE>=foundEval)
                # calculate valDiff and RPR for this time step, if there are any relevant solutions
                if np.any(relInd):
                    RPR[timeStep],valDiff[timeStep,:]=PerformIndicator.calc_RPR_one_time_step(X[relInd],foundEval[relInd],tolFunRPR,timeStep,problem)
        else: # it is a static problem
            #  select the solutions that have been found in the given evaluation budget
            timeStep=0 # not used for static problems
            startFE=0
            finishFE=problem.statData.maxEval
            relInd=np.logical_and(startFE<foundEval,finishFE>=foundEval)
            # calculate the valDiff and RPR
            RPR,valDiff=PerformIndicator.calc_RPR_one_time_step(X[relInd],foundEval[relInd],tolFunRPR,timeStep,problem)
        return RPR,valDiff

    @staticmethod
    def calc_RPR_one_time_step(X,foundEval,tolFunRPR,timeStep,problem):
        ''' calculates robust peak ratio for the static problem or one time step 
        of a dynamic problem '''

        # Output:
            # RPR: Robust Peak Ratio given the loosest and tightest function tolerances
            # valDiff: the difference between the global minimum value and the value
            # of the approximate solution found by the optimization method
        # The inputs are:
            # X (2D array): solutions reported as global minima for this time step
            # foundEval (1D array): function evaluation at which these solutions were found
            # tolFunRPR (1D array): the loosest and tightest tolerance for optimality
            # timeStep (scalar): timestep for which these solutions are considered
        problemAux=copy.deepcopy(problem) # for reevaluation of the reported solutions at reported time
        numSol=X.shape[0]
        f=np.zeros(numSol) # preallocation
        # re-evaluate the reported solution for the reported time (foundEval)
        for solNo in np.arange(numSol):
            problemAux.numCallObj=foundEval[solNo]-1
            f[solNo]=problemAux.func_eval(X[solNo,:])
        # get the global minimum value, global minima, and the niche radii from the problem object
        if  problem.isDynamic: 
            timeStepEq=np.mod(timeStep,problemAux.dynaAttr.chSevReg) # the earliest equivalent time step
            globalMinima=problem.dynaData.globMinima[timeStepEq]
            nichRad=problem.dynaData.nichRad[timeStepEq]
            fstar=problem.dynaData.globMinValOffset[timeStepEq]+problem.statData.globMinVal
        else: 
            globalMinima=problem.statData.globMinima
            nichRad=problem.statData.nichRad
            fstar=problem.statData.globMinVal
        # find the related global minimum for each reported solutions 
        belongInd=PerformIndicator.find_belong_ind(globalMinima,nichRad,X)
        # For each global minimum, calculate valDiff, the difference between the value of
        # the approximate solution (if any) and the global minimum value
        valDiff=np.ones(globalMinima.shape[0])*np.inf # preallocation
        for nichNo in np.arange(globalMinima.shape[0]):
            ind=np.where(belongInd==nichNo)[0]
            if ind.size>0: # if there is a corresponding solution for the global minimum
                valDiff[nichNo]=np.min(f[ind])-fstar
        # calculate RPR based on valDiff, which indicates how accurate each 
        # global minimum has been approximated
        l=-np.log10(np.max(tolFunRPR))
        u=-np.log10(np.min(tolFunRPR))
        tmp=np.clip((-np.log10(valDiff)-l)/(u-l),a_min=0,a_max=1)
        RPR=np.mean(tmp)
        return RPR,valDiff # outputs

    @staticmethod
    def find_belong_ind(globalMinima,Rnich,X):
        ''' find which global minimum is pertains to each reported solution '''
        belongInd=np.zeros(X.shape[0])-1
        for k in np.arange(globalMinima.shape[0]):
            dis=sp.spatial.distance.cdist(globalMinima[k,:].reshape(1,-1),X) 
            belongInd[dis.ravel()<=Rnich[k]]=k
        return belongInd

import copy
import scipy as sp
import numpy as np 
import sys  
import pickle

if __name__=='__main__':
    """ This an example to calculate RPR  the outcome of thee optimization run,
    stored in the file 'result.pk1' """
    tolFunRPR=np.array([.1,1e-5]) # loosest and tightest tolerance for optimality
    inputFile = open('result.pk1', 'rb') # results of the optimization process
    data = pickle.load(inputFile)
    # extract the required information
    foundEval=data['foundEval'] # function evaluations at which each reported solution has been found
    solution=data['solution'] # the reported solutions by the optimization method
    problem=data['problem'] # the problem object
    # now call the calc_RPR to calculate robust peak ratio and the valDiff, the 
    # difference between the global minimum value and the value of the reported solutions
    # by the optimization method
    RPR,valDiff=PerformIndicator.calc_RPR(solution,foundEval,tolFunRPR,problem)
    if problem.isDynamic:
        print('RPR at each time step: ')
        print(RPR)
    else:
        print('valDiff=',valDiff)
        print('RPR RPR =',RPR)

    
