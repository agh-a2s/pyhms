# This file runs a simple dynamic multimodal optimization (DMMO) method based on
# covariance matrix self-adaptation evolution strategy (CMSA-ES) [1], with a few
# additional termination criteria adopted from [2]. The results of this method may
# serve as a benchmark for performance comparison of the DMMO methods. After completion
# of the run, the code generates the file ‘result.pk1’ which stores the problem data,
# the reported optimal solutions, and the time (evaluation number) at which these
# solutions have been found. This file can be postprocessed to calculate the
# performance indicator.

# References c
# [1] Beyer, Hans-Georg, and Bernhard Sendhoff. "Covariance matrix adaptation revisited–the CMSA evolution strategy–."
#     In International Conference on Parallel Problem Solving from Nature, pp. 123-132. Springer, Berlin, Heidelberg, 2008.
# [2] Hansen, Nikolaus. "Benchmarking a BI-population CMA-ES on the BBOB-2009 function testbed." 
#     In Proceedings of the 11th Annual Conference Companion on Genetic and Evolutionary Computation
#     Conference: Late Breaking Papers, pp. 2389-2396. 2009.
 


def optimize_static(problem):
    ''' perform one independent restart of the optimization process and returns
    the optimized solution '''
    
    # set control parameters
    popsize=4+int(10*problem.statAttr.dim**0.5) # population size
    maxIter=100+50*(problem.statAttr.dim+3)**2 # maximum iteration 
    tolHistFun=1e-5 # desired function tolerance
    tolHistSize=int(10+30*(problem.statAttr.dim/popsize)) # condition for convergence, see [2]
    parsize=int(popsize/4) # parent size
    tau=1/(2*problem.statAttr.dim)**.5 # learning rate for the global step size
    tauC=1+problem.statAttr.dim*(1+problem.statAttr.dim)/(2*parsize)  # time interval for adaptation of the covariance matrix
    smean=0.25 # initial value of the step size
    
    # set the initial values
    xmean=(problem.statData.upBound-problem.statData.lowBound)*np.random.rand(problem.statAttr.dim)+problem.statData.lowBound
    stretch=(problem.statData.upBound-problem.statData.lowBound)
    C=np.diag(stretch**2)
    R=np.eye(problem.statAttr.dim)
    iterNo=0
    
    # preallocate the variables
    s=np.zeros(popsize) # step sizes
    x=np.zeros(([popsize,problem.statAttr.dim])) # solutions
    z=np.zeros(([popsize,problem.statAttr.dim])) # variations
    f=np.zeros(popsize) # values
    bestVals=np.random.rand(tolHistSize) # best value at each iteration
    while iterNo<maxIter:
        iterNo+=1
        # form and evaluate the new population
        for solNo in np.arange(popsize):
            s[solNo]=smean*np.exp(np.random.randn()*tau)
            z[solNo]=np.dot(R,stretch*np.random.randn(problem.statAttr.dim))
            x[solNo]=xmean+z[solNo]*s[solNo]
            f[solNo]=problem.func_eval(x[solNo])
        # perform non-elite selection
        ind=np.argsort(f)
        # update the population center
        xmean= np.mean(x[ind[0:parsize]],axis=0)
        # update the mutation profile
        smean= np.mean(s[ind[0:parsize]]) 
        suggC=np.zeros((problem.statAttr.dim,problem.statAttr.dim))
        for parNo in np.arange(parsize):
            tmp=z[ind[parNo]].reshape(1,-1)
            suggC+=np.dot(tmp.transpose(),tmp)/parsize
        
        C=(1-1.0/tauC)*C+(1.0/tauC)*suggC
        C=0.5*(C+C.transpose()) # enforce symmetry of C
        tmp,R=sp.linalg.eigh(C) # perform eigen decomposition to calculate rotMat and stretch
        stretch=np.real(np.sqrt(tmp))
        # update the history of best values
        bestVals[:-1]=bestVals[1:]
        bestVals[-1]=f[ind[0]]
        
        foundEval=problem.numCallObj # the evaluation at which the final solution has been found
        #print(foundEval,bestVals[-1])
        # check termination criteria
        if (np.max(bestVals)-np.min(bestVals))<tolHistFun or (np.max(stretch)/np.min(stretch))>1e7:
            break
    # return the final solution, its value, and the evaluation at which it was found
    return foundEval,x[ind[0]].copy(), f[ind[0]],iterNo

 
def optimize_full(problem):
    # Initialize reported solutions and the time they have been found
    solution=np.zeros((0,problem.statAttr.dim))
    foundEval=np.zeros(0)
    restartNo=-1
    # perform independent restart until the evaluation budget has been used
    while problem.numCallObj<problem.maxEvalTotal:
        restartNo+=1
        # perform one independent restart and store the final solution and the evaluation at which it was found
        fe,newX,newF,usedIter=optimize_static(problem)
        print('Restart #'+str(restartNo)+',','FE='+str(problem.numCallObj)+'('+'{:.3}'.format(problem.numCallObj/problem.maxEvalTotal*100)+'%),','f_best(x)='+str(newF))
        # keep track of the found solutions
        foundEval=np.concatenate((foundEval,np.array([fe])))
        solution=np.concatenate((solution,newX.reshape(1,-1)),axis=0)
    return foundEval, solution


from DDRB import DDRB
from PerformIndicator import PerformIndicator
import numpy as np 
import scipy as sp
import pickle 

np.random.seed(0) # random seed number

PID=1 # problem ID
dynaScn=6 # dynamic scenario
problem=DDRB(PID,dynaScn)  # creates a problem object that stores both static and dynamic attributes  
# you can change the problem attributes before loading the problem data
problem.statAttr.dim=8
problem.dynaAttr.numTimeStep=10


# calculate the problem data (no more change in the problem attributes is allowed afterwards)
problem.calc_problem_data()

# perform optimization
print("\nPerforming static/dynamic multimodal optimization ...")
foundEval, solution=optimize_full(problem) 

print("\nOptimization completed. Calculating the Performance ...\n")
# calculate the performance
tolFunScore=np.array([0.1,0.00001])
RPR,valDiff=PerformIndicator.calc_RPR(solution,foundEval,tolFunScore,problem)

# display the measured performance
print('RPR for each time step is \n ',RPR )
print('\nDifference between the value of the approximate solution and the global minimum value for each global minimum at each time step is:')
print(valDiff)

