
''' The object from this class defines the optimization problem.
For queries, please contact Ali Ahrari at aliahrari1983@gmail.com or a.ahrari@unsw.edu.au '''

class DDRB:
    __slots__=['statAttr','statData','dynaAttr','dynaData','maxEvalTotal','isDynamic','numCallObj']
    #Properties:
        # statAttr (object): attributes related to static aspects of problem (can be changed by the user)
        # statData (object): data related to static problem or the zeroth time step of a dynamic problem (determined by this class)
        # dynaAttr (object): attributes related to the dynamic aspects of problem (can be changed by the user)
        # dynaData (object): data related to the dynamic problem (determined by this class)
        # maxEvalTotal (int): total number of function evaluations (determined by this class)
        # isDynamic (bool): is the problem dynamic?
        # numCallObj (int) (counter for the number of calls to the objective function)

    def __init__(self,PID,dynaScn): 
        ''' This is the class constructor '''
        # PID (integer between 1 and 10 inclusive): ID of the composite function 
        # dynaScn: (integer between 0 and 6, inclusive): dynamic scenario (if 1-6). For dynaScn=0, the problem is static 
        pidInfo=np.array([[11,12,13,14,15,11,12,13,14,15],[2,2,2,1,2,4,6,4,2,4]]) # problem ID's (function ID and and D_I)
        scnInfo=[[0.3,0.5,2000,30,np.inf],
                [0.8,0.5,2000,30,np.inf],
                [0.3,0.1,2000,30,np.inf],
                [0.3,0.5,2000,10,np.inf],
                [0.3,0.5,2000,30,5],
                [0.3,0.5,500,30,np.inf]] # dynamic Scenrios
        
        # extract problem attributes based on the selected PID and scenario
        dim=10  # problem dimensionality
        maxEvalStatCoeff=4000 # coefficient for the evaluation budget of the static problem (or the zeroth time step if the problem is dynamic)
        funcID=pidInfo[0,PID-1] # function ID (index of G function) (not the problem ID)
        D_I=pidInfo[1,PID-1] # D_I (determines the number of global minima)
        if dynaScn==0: # if the problem is static 
            h_GO,e_c,chFrCoeff,chSevReg,chSevIrreg=scnInfo[0] # attributes of the problem based on the predefined dynamic scenarios 
            self.isDynamic=False # problem is not dynamic
            numTimeStep=1 # only one time step if the problem is static
        elif dynaScn>=1 and dynaScn<=6: # if problem is dynamic
            self.isDynamic=True
            h_GO,e_c,chFrCoeff,chSevReg,chSevIrreg=scnInfo[dynaScn-1] # attributes of the problem based on the predefined dynamic scenarios 
            # h_GO: global optimization hardness of the problem (0<=h_GO<=1)
            # e_c: eccentricity for distortion (0<e_c<inf)
            # chFrCoeff: coefficient for the change frequency (chFrCoeff>0)
            # chSevReg: severity of the regular change (chSevReg>0)
            # chSevIrreg: severity of the irregular change (chSevIrreg>0)
            numTimeStep=40 # the number of time steps

        else:  
            sys.exit('dynaScn must be an int between 0 and 6, inclusive')
            
        self.statAttr=StaticAttribute(funcID,D_I,dim,h_GO,maxEvalStatCoeff) # form static attributes
        self.dynaAttr=DynamicAttribute(chSevReg,chSevIrreg,chFrCoeff,e_c,numTimeStep)  # form dynamic attributes 
        self.statData=StaticData() # static data (will be determined later automatically)
        self.dynaData=DynamicData() # dynamic data (will be determined later automatically)
        self.maxEvalTotal='N/A' # will be determined later automatically
        self.numCallObj=int(0) # will be determined later automatically

    def __str__(self):  
        ''' This method displays the status of the object '''
        output= ('\n  maxEvalTotal  = ' + str (self.maxEvalTotal ) +  
                 '\n  isDynamic  = ' + str (self.isDynamic ) +
                 '\n  numCallObj  = ' + str (self.numCallObj ) +
                 '\n  statAttr.   ' + self.statAttr.__str__() +  
                 '\n  statData. ' +  self.statData.__str__() +  
                 '\n  dynaAttr. ' + self.dynaAttr.__str__() +  
                 '\n  dynaData. ' + self.dynaData.__str__()   )  
        return output
    
    def calc_problem_data(self): 
        ''' calculate problem data and store them '''
        # check if the static and dynamic attributes are valid
        numErrStat=self.statAttr.check_validity()
        numErrDyna=0
        if self.isDynamic:
            numErrDyna=self.dynaAttr.check_validity()
        
        if numErrStat+numErrDyna>0:
            print('There are ' + str(numErrStat+numErrDyna) + ' properties of statAttr and/or dynaAttr with invalid values. Please correct them and try again.')
            sys.exit()
        else: # Static and dynamic properties have valid values, proceed to calculate the problem data
            print('Parameters of the problem are valid. Calculating problem data ...',end='')
            self.statData.calc_static_data(self.statAttr) # calculate and provide data for the static aspects of the problem
            self.maxEvalTotal=self.statData.maxEval
            if self.isDynamic: # if problem is dynamic
                self.dynaData.calc_dynamic_data(self.statAttr,self.dynaAttr,self.statData) # calculate and provide data for the static aspects of the problem
                self.maxEvalTotal=self.statData.maxEval+(self.dynaAttr.numTimeStep-1)*self.dynaData.chFr # total evaluation budget of a dynamic problem
            print(' Done.')


    def func_eval(self,x_ini):
        ''' calculate the objective value for the solution x_ini '''
        # Inputs: 
            # x_ini: solution to be evaluated. It is a vector with D elements (matrix input is not accepted) 
            # self: object from this class
        # Output: 
            # objVal: The solution value (objective function value for solution x)
        self.numCallObj+=1 # update this attribute
        x=x_ini.copy()
        if self.numCallObj>self.maxEvalTotal:
            objVal=1.7e308 # beyond the budget, a very large value
        else:
            dynaValOffs=0 # offset defined at each time step for the dynamic problem
            if  self.isDynamic:  
                timeStep=int(np.clip(np.ceil((self.numCallObj-self.statData.maxEval)/
                            self.dynaData.chFr),a_min=0, a_max=np.inf))  # calculate the time step 
                timeStepEq=np.mod(timeStep,self.dynaAttr.chSevReg) # the earliest equivalent time step, since the landscape repeats after each chFr evalautions 
                x=ScaleFunc.scale_it(x,self.dynaData.scaleParX[timeStepEq],
                                     self.statData.lowBound,self.statData.upBound,self.dynaAttr.e_c) # perform the scaling of the search space
                if self.dynaAttr.performDynaRot: # if dynamic rotation is performed
                    x=np.dot(x,self.dynaData.rotMat[timeStepEq].transpose())  # dynamic rotation equal to (R(t)*X')'
                dynaValOffs=self.dynaData.globMinValOffset[timeStepEq] # offset defined at each time step for the dynamic problem
            # no change to x if the problem is static 
            # now x has been scaled and rotated, the objective function value can be calculated as follows:
            objVal=DDRB.func_eval_static(x,self.statAttr,self.statData)+dynaValOffs
        return objVal
         
    
    @staticmethod
    def func_eval_static(X,statAttr,statData): # static multimodal function
        ''' cacluates the objective value of the static composite function '''
        # Inputs:
            # X (vector of size D): solution at which the objective function is calculated
            # statAttr (object): static attributes of the problem class 
            # statData (object): static features of the problem class
        # Outputs:
            # f: value of solution X
        X=np.dot(X,statData.rotMat.transpose()) # static rotation equal to (R_0*X')'
        if statAttr.funcID==11: # static composite function G11(x,D_I,h_GO) 
            y1=BasicFunc.himl(X[0:statAttr.D_I])
            y2=BasicFunc.griewank(X[statAttr.D_I:],statAttr.h_GO)
            if np.isscalar(y1):
                y1=np.array([y1])
            if np.isscalar(y2):
                y2=np.array([y2])
            Y=np.concatenate((y1,y2))
            f=BasicFunc.zakharov(Y)+statData.globMinVal
 
        elif statAttr.funcID==12:  # static composite function G12(x,D_I,h_GO)
            y1=BasicFunc.hump6(X[0:statAttr.D_I])
            y2=BasicFunc.lvn13(X[statAttr.D_I:],statAttr.h_GO)
            if np.isscalar(y1):
                y1=np.array([y1])
            if np.isscalar(y2):
                y2=np.array([y2])
            Y=np.concatenate((y1,y2))                
            f=np.linalg.norm(Y)**2+statData.globMinVal  
                
        elif statAttr.funcID==13: # static composite function G13(x,D_I,h_GO)
            y1=BasicFunc.branin(X[0:statAttr.D_I])
            y2=BasicFunc.hump3(X[statAttr.D_I:],statAttr.h_GO)
            if np.isscalar(y1):
                y1=np.array([y1])
            if np.isscalar(y2):
                y2=np.array([y2])
            Y=np.concatenate((y1,y2))
            f=BasicFunc.neumaier3(Y)+statData.globMinVal
 
        elif statAttr.funcID==14:  # static composite function G14(x,D_I,h_GO)
            y1=BasicFunc.shubert(X[0:statAttr.D_I])
            y2=BasicFunc.weierstrass(X[statAttr.D_I:],statAttr.h_GO)
            if np.isscalar(y1):
                y1=np.array([y1])
            if np.isscalar(y2):
                y2=np.array([y2])
            Y=np.concatenate((y1,y2))
            f=BasicFunc.booth(Y)
            
        elif statAttr.funcID==15:  # static composite function G15(x,D_I,h_GO)
            y1=BasicFunc.cmmp(X[0:statAttr.D_I])
            y2=BasicFunc.bohach2(X[statAttr.D_I:],statAttr.h_GO)
            if np.isscalar(y1):
                y1=np.array([y1])
            if np.isscalar(y2):
                y2=np.array([y2])
            Y=np.concatenate((y1,y2))
            f=BasicFunc.dp(Y)**2+statData.globMinVal  
              
        elif statAttr.funcID==16:  # A simple problem for illustration
            f=-np.mean(np.cos(np.pi*X))

        else:
            sys.exit('Error in DDRB.func_eval_static: this function has not been defined')

        return f
        
class DynamicData:
    ''' the object from this class stores data related to dynamic spects of the problem '''
    __slots__=['rotMat','globMinima','globMinValOffset','scaleParX','chFr','nichRad']
    def __init__(self): 
        ''' This is the class constructor '''
        self.rotMat=np.zeros((0,0,0)) # (3D array) rigid rotation matrix at all time steps (x1 is time step, x2 and x3 store the matrix for a specific time step)  
        self.globMinima=np.zeros((0,0,0)) # (3D array) global minima at all time steps (x1 is time step, x2 is a global minimum, x3 is a component of the global minimum) 
        self.globMinValOffset=np.zeros(0) # (1D array) shift in the global minimum value for all time steps. It is c(t) in equation 13. 
        self.scaleParX=np.zeros(0) # (1D array)  values of the scaling parameter (w(t)) for all time steps.   
        self.chFr='N/A' # change frequency (scalar)
        self.nichRad=np.zeros((0,0)) # (2D array) the niche radii of all global minima for all time steps

    def __str__(self):  
        ''' This method displays the status of the object '''
        output= ('\n\trotMat is an array of shape ' + str (self.rotMat.shape) +  
                 '\n\tglobMinima is an array of shape ' + str (self.globMinima.shape) +  
                 '\n\tglobMinValOffset = ' + str (self.globMinValOffset) +  
                 '\n\tscaleParX = ' + str (self.scaleParX) + 
                 '\n\tchFr = ' + str (self.chFr) +  
                 '\n\tnichRad is an array of shape ' + str (self.nichRad.shape) ) 
        return output

    def calc_dynamic_data(self,statAttr,dynaAttr,statData):
        dataSize=np.min((dynaAttr.chSevReg,dynaAttr.numTimeStep))
        # note that since after chSevReg changes the problem landscape returns to its
        # status at chSevReg time steps before, it is sufficient to store the data for time steps 0,1,2,..., dataSize
        # index 0 stores data for the first time step, which are data of the static problem
        
        # Calculate and store the problem features if it is dynamic
        tmp=np.arange(0,dataSize)
        rotAngle=2*np.pi*( tmp/dynaAttr.chSevReg + np.sin(tmp**2)/dynaAttr.chSevIrreg)
        self.scaleParX=np.sin(2*np.pi*tmp/dynaAttr.chSevReg + 2*np.pi*np.sin(tmp**2)/dynaAttr.chSevIrreg) 
        self.globMinValOffset=100*np.sin(2*np.pi*tmp/dynaAttr.chSevReg + 2*np.pi*np.sin(tmp**2)/dynaAttr.chSevIrreg)
        
        # form the rotation matrixes of all time steps        
        self.rotMat= np.zeros((dataSize,statAttr.dim,statAttr.dim)) # preallocation  
        for timeStep in np.arange(dataSize):
            self.rotMat[timeStep,:,:]=AuxiliaryMethods.gen_rot_mat(statAttr.dim,rotAngle[timeStep]) # rotation matrix for each time step 
 
        del timeStep
        # calculate the location of global minima over time
        self.globMinima= np.zeros((dataSize,statData.numGlobMin,statAttr.dim)) # preallocation
        for timeStep in np.arange(dataSize):
            if dynaAttr.performDynaRot: # if rotation is performed, consider it on the locations of global minima
                self.globMinima[timeStep,:,:]=np.dot(statData.globMinima,self.rotMat[timeStep]) # equal to (R(t)'*X')'
            else:
                self.globMinima[timeStep,:,:]=statData.globMinima.copy()

            for minNo in np.arange(statData.numGlobMin): # apply the effect of scaling the search space on the locations of global minima 
                self.globMinima[timeStep,minNo,:]=ScaleFunc.scale_it_inv(self.globMinima[timeStep][minNo,:],self.scaleParX[timeStep],statData.lowBound,statData.upBound,dynaAttr.e_c)

        # calculation of the niche radii of each global minimum at each time step
        self.nichRad=np.zeros((dataSize,statData.numGlobMin)) # preallocation
        for timeStep in np.arange(dataSize):
            self.nichRad[timeStep]=np.inf*np.ones(statData.numGlobMin) # preallocation
            if statData.numGlobMin>1: # if there are multiple global minima
                for minNo in np.arange(statData.numGlobMin):
                    thisSol=self.globMinima[timeStep,minNo,:].reshape(-1,statAttr.dim)
                    otherSols=self.globMinima[timeStep,np.arange(statData.numGlobMin)!=minNo,:].reshape(-1,statAttr.dim) # distance to other global minima
                    self.nichRad[timeStep,minNo]=0.5*np.min(cdist(thisSol,otherSols)) # half of the distance to the closest global minimum
        # set the change frequency
        self.chFr=dynaAttr.chFrCoeff*statAttr.dim*statData.numGlobMin # 

class StaticData:
    ''' An object from this class will store the calculated data for the static 
    problem. These data will be calculated by the method calc_static_data(self,statAttr). '''
    
    __slots__=['globMinima','globMinVal','rotMat','numGlobMin','maxEval','nichRad','lowBound','upBound']
    def __init__(self): 
        ''' This is the class constructor '''
        self.globMinima=np.zeros((0,0))  # (2D array) global minima of the static problem 
        self.globMinVal='N/A' # (scalar) the global minimum value of the static problem 
        self.rotMat=np.zeros((0,0))   # (matrix) rotation matrix of the static problem 
        self.numGlobMin='N/A' # (scalar) number of global minima
        self.maxEval='N/A' # (scalar) evaluation budget of the static problem (or for the time step #0 if the problem is dynamic)
        self.nichRad='N/A' # (scalar) the niche radius (for postprocessing only)
        self.lowBound=np.zeros(0) # (vector) the lower bound of the search space
        self.upBound=np.zeros(0) # (vector) the upper bound of the search space
    
    def __str__(self):  
        ''' This method displays the status of the object '''
        output= ('\n\tglobMinima is a 2D array of shape ' + str (self.globMinima.shape) +  
                 '\n\tglobMinVal = ' + str (self.globMinVal) + 
                 '\n\trotMat is a 2D array of shape ' + str (self.rotMat.shape) + 
                 '\n\tnumGlobMin = ' + str (self.numGlobMin) + 
                 '\n\tmaxEval = ' + str (self.maxEval) + 
                 '\n\tnichRad = ' + str (self.nichRad) +
                 '\n\tlowBound = ' + str (self.lowBound) +
                 '\n\tupBound = ' + str (self.upBound) )
        return output
    
    def calc_static_data(self,statAttr):
        ''' This method calculates the data related to the static features of the problem and
        then update statData of the problem object ''' 
        
        # define the global minimum value and basicGlobMin, the global minima of 
        # the subfunction g_I for each composite function
        if statAttr.funcID==11: # static composite function G11(x,D_I,h_GO) 
            self.globMinVal=-49.2 
            basicGlobMin=np.transpose(np.array([   
                         [3,	-2.80511808695274,	-3.77931025337774,	3.58442834033049],
                         [2,     3.13131251825057,	-3.28318599128616,	-1.84812652696440]   
                         ])/3)
            
        elif statAttr.funcID==12: # # static composite function G12(x,D_I,h_GO)      
            self.globMinVal=-29.7
            basicGlobMin=np.transpose(np.array([      
                         [0.081674553779591,	-0.099824459882615],
                         [-0.647869456984275,	0.791840448995349],
                         ])) 
            
        elif statAttr.funcID==13: # # static composite function G13(x,D_I,h_GO) 
            self.globMinVal=95.1
            basicGlobMin=np.transpose(np.array([      
                         [1.606194492,	0.035398163,	-1.535398165],
                         [-1.131250003,	-1.181249999,	1.318750004]
                         ])) 
            
        elif statAttr.funcID==14:   # # static composite function G14(x,D_I,h_GO) 
            self.globMinVal=0
            if statAttr.D_I==3: # if D_I==3
                basicGlobMin=np.transpose(np.array([  
                                [-7.083506408,-0.8003211,5.482864206,-7.083506408,-0.8003211,5.482864206,-7.083506408,-0.8003211,5.482864206,-7.083506408,-0.8003211,5.482864206,-7.708313736,-1.425128428,4.85805688,-7.083506408,-0.8003211,5.482864206,-7.708313736,-1.425128428,4.85805688,-7.083506408,-0.8003211,5.482864206,-7.708313736,-1.425128428,4.85805688,-7.083506408,-0.8003211,5.482864206,-7.083506408,-0.8003211,5.482864206,-7.083506408,-0.8003211,5.482864206,-7.083506408,-0.8003211,5.482864206,-7.708313736,-1.425128428,4.85805688,-7.083506408,-0.8003211,5.482864206,-7.708313736,-1.425128428,4.85805688,-7.083506408,-0.8003211,5.482864206,-7.708313736,-1.425128428,4.85805688,-7.083506408,-0.8003211,5.482864206,-7.083506408,-0.8003211,5.482864206,-7.083506408,-0.8003211,5.482864206,-7.083506408,-0.8003211,5.482864206,-7.708313736,-1.425128428,4.85805688,-7.083506408,-0.8003211,5.482864206,-7.708313736,-1.425128428,4.85805688,-7.083506408,-0.8003211,5.482864206,-7.708313736,-1.425128428,4.85805688],
                                [-7.083506408,-7.083506408,-7.083506408,-0.8003211,-0.8003211,-0.8003211,5.482864206,5.482864206,5.482864206,-7.708313736,-7.708313736,-7.708313736,-7.083506408,-7.083506408,-7.083506408,-1.425128428,-1.425128428,-1.425128428,-0.8003211,-0.8003211,-0.8003211,4.85805688,4.85805688,4.85805688,5.482864206,5.482864206,5.482864206,-7.083506408,-7.083506408,-7.083506408,-0.8003211,-0.8003211,-0.8003211,5.482864206,5.482864206,5.482864206,-7.708313736,-7.708313736,-7.708313736,-7.083506408,-7.083506408,-7.083506408,-1.425128428,-1.425128428,-1.425128428,-0.8003211,-0.8003211,-0.8003211,4.85805688,4.85805688,4.85805688,5.482864206,5.482864206,5.482864206,-7.083506408,-7.083506408,-7.083506408,-0.8003211,-0.8003211,-0.8003211,5.482864206,5.482864206,5.482864206,-7.708313736,-7.708313736,-7.708313736,-7.083506408,-7.083506408,-7.083506408,-1.425128428,-1.425128428,-1.425128428,-0.8003211,-0.8003211,-0.8003211,4.85805688,4.85805688,4.85805688,5.482864206,5.482864206,5.482864206],
                                [-7.708313736,-7.708313736,-7.708313736,-7.708313736,-7.708313736,-7.708313736,-7.708313736,-7.708313736,-7.708313736,-7.083506408,-7.083506408,-7.083506408,-7.083506408,-7.083506408,-7.083506408,-7.083506408,-7.083506408,-7.083506408,-7.083506408,-7.083506408,-7.083506408,-7.083506408,-7.083506408,-7.083506408,-7.083506408,-7.083506408,-7.083506408,-1.425128428,-1.425128428,-1.425128428,-1.425128428,-1.425128428,-1.425128428,-1.425128428,-1.425128428,-1.425128428,-0.8003211,-0.8003211,-0.8003211,-0.8003211,-0.8003211,-0.8003211,-0.8003211,-0.8003211,-0.8003211,-0.8003211,-0.8003211,-0.8003211,-0.8003211,-0.8003211,-0.8003211,-0.8003211,-0.8003211,-0.8003211,4.85805688,4.85805688,4.85805688,4.85805688,4.85805688,4.85805688,4.85805688,4.85805688,4.85805688,5.482864206,5.482864206,5.482864206,5.482864206,5.482864206,5.482864206,5.482864206,5.482864206,5.482864206,5.482864206,5.482864206,5.482864206,5.482864206,5.482864206,5.482864206,5.482864206,5.482864206,5.482864206]
                             ])) 
            elif statAttr.D_I==2: # if D_I==2
                basicGlobMin=np.transpose(np.array([ 
                                [-7.083506408,-0.8003211,5.482864206,-7.708313736,-1.425128428,4.85805688,-7.083506408,-0.8003211,5.482864206,-7.708313736,-1.425128428,4.85805688,-7.083506408,-0.8003211,5.482864206,-7.708313736,-1.425128428,4.85805688],
                                [-7.708313736,-7.708313736,-7.708313736,-7.083506408,-7.083506408,-7.083506408,-1.425128428,-1.425128428,-1.425128428,-0.8003211,-0.8003211,-0.8003211,4.85805688,4.85805688,4.85805688,5.482864206,5.482864206,5.482864206]
                             ])) 
            elif statAttr.D_I==1: # if D_I==1
                basicGlobMin=np.transpose(np.array([ 
                              [-7.708313736,4.85805688,-1.425128428]
                             ])) 
            basicGlobMin=basicGlobMin/4
            
        elif statAttr.funcID==15:  # # static composite function G15(x,D_I,h_GO) 
            self.globMinVal=65.5
            a=np.sqrt(27/7);
            b=np.sqrt(4/7);
            basicGlobMin=np.transpose(np.array([
                         [a,  -a,    a, -a,],
                         [b,   b,   -b, -b,]
                         ])) 
        elif statAttr.funcID==16: # a simple test problem
            self.globMinVal=0 
            basicGlobMin=np.transpose(np.array([[-2, 0, 2]])) 
        else:
            sys.exit('error in DDRB.calc_static_data(self,statAttr). This problem has not been defined')
        
        # set the search range
        self.lowBound=-5*np.ones(statAttr.dim) # lower bound of the search space
        self.upBound=5*np.ones(statAttr.dim)  # upper bound of the search space
            
        # calculate global minima of the static composite function given the global minima of the first basic function (G_I)
        Nrep=int(statAttr.D_I/basicGlobMin.shape[1]+.5) # D_I/d_i
        self.numGlobMin=basicGlobMin.shape[0]**Nrep # number of global minima of the composite function
        ind=AuxiliaryMethods.gen_rep_ind(Nrep,basicGlobMin.shape[0]) # an auxiliary method used to determine the locations of the global minima
        indSize=ind.shape[0]
        tmpGlobMinima=np.zeros((self.numGlobMin,statAttr.D_I)) # preallocation
        for k1 in np.arange(indSize):
            for k2 in np.arange(Nrep):
                tmpGlobMinima[k1,k2*basicGlobMin.shape[1]+np.arange(basicGlobMin.shape[1])]=basicGlobMin[ind[k1,k2],:]
        self.globMinima=np.concatenate((tmpGlobMinima, np.zeros((indSize,statAttr.dim-statAttr.D_I))),axis=1) # global minima of the static composite function before applying rotation 
        # Generate the rotation matrix for the static problem
        self.rotMat=AuxiliaryMethods.gen_rot_mat(statAttr.dim,statAttr.rotAngle) # generate the rigid rotation matrix given the rotation angle 
        self.globMinima=np.dot(self.globMinima,self.rotMat) # global minima of the rotated composite function, equal to (R_0'*X')'
        # calculate the maximum evalaution budget for the static problem
        self.maxEval=statAttr.maxEvalCoeff*statAttr.dim*self.numGlobMin;
        # calculatie the niche radius for each global minimum (static problem)
        self.nichRad=np.inf*np.ones(self.numGlobMin) # preallocation
        if self.numGlobMin>1:
            indexes=np.arange(self.numGlobMin)
            for minNo in indexes: # for each global minimum
                # calculate the distance between this global minimum and the closest one
                thisSol=self.globMinima[minNo].reshape(1,-1)
                otherSols=self.globMinima[indexes!=minNo].reshape(-1,statAttr.dim)
                # set the corresponding niche radius equal to half of this distance
                self.nichRad[minNo]=0.5*np.min(cdist(thisSol,otherSols)) # the niche radius of global minimum minNo

class DynamicAttribute:
    '''An object from this class stores the dynamic attributes of the problem'''
    __slots__=['chSevReg','chSevIrreg','chFrCoeff','e_c','numTimeStep','performDynaRot']
    
    def __init__(self,chSevReg,chSevIrreg,chFrCoeff,e_c,numTimeStep): 
        ''' This is the class constructor '''
        self.chSevReg=chSevReg  # change severity parameter (regular)
        self.chSevIrreg=chSevIrreg # change severity parameter (irregular) 
        self.chFrCoeff=chFrCoeff # change frequency coefficient (it will be multiplied by D*N_gmin to calculate the change frequency)
        self.e_c=e_c # eccentricity for the scaling function (0<e_c) for dynamic distortion of the landscape
        self.numTimeStep=numTimeStep # number of time steps 
        self.performDynaRot=True # if the landscape rotates over time (it should be true by default) 
            
    def check_validity(self):
        '''check for the validity of the parameters determining the dynamic features of the problem'''
        numErr=0
        # check chSevReg
        if self.chSevReg<=0:
            print('Error: The value of dynaAttr.chSevReg is not valid. It must be greater than zero.')
            numErr+=1
        # check chSevIrreg
        if self.chSevIrreg<=0:
            print('Error: The value of dynaAttr.chSevIrreg is not valid. It must be greater than zero.')
            numErr+=1
        # check chFrCoeff
        if self.chFrCoeff<=0:
            print('Error: The value of dynaAttr.chFrCoeff is not valid. It must be greater than zero.')
            numErr+=1   
        # check e_c
        if self.e_c<=0:
            print('Error: The value of dynaAttr.e_c is not valid. It must be greater than zero.')
            numErr+=1            
        # check numTimeStep
        if self.numTimeStep<=0 or (not isinstance(self.numTimeStep,int) ):
            print('Error: The value of dynaAttr.numTimeStep is not valid. It must be a positive integer.')
            numErr+=1 
        # check numTimeStep
        if not isinstance(self.performDynaRot,bool):
            print('Error: The value of dynaAttr.performDynaRot is not valid. It must be False or True.')
            numErr+=1             
            
            
            
        return numErr # number of errors   
            
    def __str__(self):  
        ''' This method displays the status of the object '''
        output= ('\n\tchSevReg = ' + str (self.chSevReg) +  
                 '\n\tchSevIrreg = ' + str (self.chSevIrreg) + 
                 '\n\tchFrCoeff = ' + str (self.chFrCoeff) + 
                 '\n\te_c = ' + str (self.e_c) + 
                 '\n\tnumTimeStep = ' + str (self.numTimeStep) + 
                 '\n\tperformDynaRot = ' + str (self.performDynaRot)  )
        return output

class StaticAttribute:
    ''' An object from this class stores the static attributes of the problem '''
    __slots__=['funcID','D_I','dim','h_GO','maxEvalCoeff','rotAngle']
    
    def __init__(self,funcID,D_I,dim,h_GO,maxEvalCoeff): 
        ''' This is the class constructor '''
        self.funcID=funcID # index of the function (11, 12, 13, 14 , or 15)
        self.D_I=D_I # function parameter that controls the number of global minima 
        self.dim=dim # search space dimensionality
        self.h_GO=h_GO # function parameter that controls the difficulty of global optimization
        self.maxEvalCoeff=maxEvalCoeff # the coefficient for the evaluation budget of the static problem. It is the evaluation budget divided by (dim*N_gmin)
        self.rotAngle=np.pi/6 # the rotation angle of the employed rotation matrix for the static problem
        
    def check_validity(self):
        '''check for the validity of the parameters determining the static features of the problem'''
        numErr=0 # number of mistakes in parameter setting made by the user
        # check the validity of problem dimensionality    
        if np.isin(self.funcID,np.array([11,12,13,15])):
            k=(self.dim-self.D_I)
            if not (np.mod(k,2)==0 and k>=0 and isinstance(self.dim,int)):
                print('Error: The problem dimensionality (statAttr.dim) is not valid. For this problem, valid options are 2k+'+str(self.D_I)+', in which k is a non-negative integer.')
                numErr+=1
            elif self.funcID==14:
                k=(self.dim-self.D_I)
                if not (k>=0 and isinstance(self.dim,int)):
                    print('Error: The problem dimensionality (statAttr.dim) is not valid. For this problem, valid options are k+'+str(self.D_I)+', in which k is a non-negative integer.')
                numErr+=1
        # check h_GO
        if self.h_GO<0 or self.h_GO>1:
            print('Error: The value of statAttr.h_GO is not valid. It must be between 0 and 1, inclusive.')
            numErr+=1
        # check maxEvalCoeff
        if self.maxEvalCoeff<=0:
            print('Error: The value of statAttr.maxEvalCoeff is not valid. It must be a positive number.')
            numErr+=1       
        return numErr # the number of errors
    
    def __str__(self):  
        ''' This method displays the status of the object '''
        output= ('\n\tfuncID = ' + str (self.funcID) +  
                 '\n\tD_I = ' + str (self.D_I) + 
                 '\n\tdim = ' + str (self.dim) + 
                 '\n\th_GO = ' + str (self.h_GO) + 
                 '\n\tmaxEvalCoeff = ' + str (self.maxEvalCoeff) + 
                 '\n\trotAngle = ' + str(self.rotAngle) )
        return output
    
class ScaleFunc:   
    ''' This class has only two static methods in this class are used for distortion of the search space '''
    @staticmethod
    def scale_it(x,w_t,LX,UX,e_c): # performs nonlinear scaling 
        # x is the solution (horizontal vector)
        # w_t is the scaling parameter (w(t)) in equation 13
        # LX and UX are the search range (horizontal vectors)
        # e_c is eccentricity
        # output is the image of x in the scaled space (horizontal vector)

        r=(UX-LX)/2
        center=(UX+LX)/2;
        xNorm=((x-center)/r) 
        w=abs(w_t) 
        if w_t>=0:
            term1=(e_c+1)**2 + e_c**2 - (np.abs(xNorm)-(e_c+1))**2
            y1Norm = np.sign(xNorm)*(np.sqrt( np.clip(term1,a_min=0,a_max=np.inf) )  - e_c)
        else:
            tmp=  (e_c+1)**2 + e_c**2 - (np.abs(xNorm)+e_c)**2
            y1Norm= np.sign(xNorm)*(-np.sqrt( np.clip(tmp,a_min=0,a_max=np.inf) )+(1+e_c))

        yNorm= (1-w)*xNorm + w*y1Norm
        y = yNorm*r+center
        return y

        
    @staticmethod
    def scale_it_inv(y,w_t,LX,UX,e_c): # performs the inverse scaling
        # see the previous function in this class for the definition of each argument
        r=(UX-LX)/2
        center=(UX+LX)/2
        yNorm=((y-center)/r)
        y=np.abs(yNorm)
        w=np.abs(w_t)
        if w_t>=0:
            term1=e_c**2 - 4*e_c*w*y + 2*e_c*w + 2*e_c*y + w**2 - 2*w*y - y**2 + 2*y
            xNorm =(y + e_c*w - w*y - w*(  np.clip(term1,a_min=0,a_max=np.inf)  )**(1/2) + w**2)/(2*w**2 - 2*w + 1)
            xNorm=xNorm*np.sign(yNorm)
        else: 
            term1=e_c**2 + 4*e_c*w*y - 2*e_c*w - 2*e_c*y + 2*e_c + w**2 + 2*w*y - 2*w - y**2 + 1
            xNorm= -(w - y + e_c*w + w*y - w*( np.clip(term1,a_min=0,a_max=np.inf) )**(1/2) - w**2)/(2*w**2 - 2*w + 1)
            xNorm=xNorm*np.sign(yNorm)
        x = xNorm*r+center
        return x

class AuxiliaryMethods:
    ''' This class has some handy methods '''
    @staticmethod    
    def gen_rep_ind(Nrep,Ng): # make an array of indexes for repetition of global minima
        rowNo=-1
        ind=np.zeros(Nrep,dtype=int)
        IND=np.zeros((Ng**Nrep,Nrep),dtype=int)
        while rowNo<(Ng**Nrep-1):
            rowNo=rowNo+1;
            for n in np.arange(1,Nrep+1):
                ind[n-1]= np.floor(np.mod(rowNo,Ng**n)/Ng**(n-1))
            IND[rowNo,:]=ind.copy()
        return IND
    
    @staticmethod    
    def gen_rot_mat(D,alp): # Generates a random rotation matrix 
        num=np.loadtxt('mmop/normRandNum.csv') # import random numbers
        u0=num[0:D].reshape(-1,1)
        v0=num[D:(2*D)].reshape(-1,1)
        u=u0/np.linalg.norm(u0)
        v=v0-np.dot(u.transpose(),v0)*u
        v=v/np.linalg.norm(v)
        term1=np.dot(v,u.transpose())-np.dot(u,v.transpose())
        term2=np.dot(u,u.transpose())+np.dot(v,v.transpose())
        R=np.eye(D)+np.sin(alp)*term1+(np.cos(alp)-1)*term2
        return R
    
import numpy as np
from scipy.spatial.distance import cdist
from .BasicFunc import BasicFunc
import sys

if __name__=='__main__':
    problem=DDRB(1,4) # create the problem object
    problem.calc_problem_data() # calculate and store data required for problem
    print(problem)
    y=problem.func_eval(np.random.rand(problem.statAttr.dim)) # calculate the value of a random solution
    
    
    