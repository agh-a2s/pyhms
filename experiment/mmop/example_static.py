''' An example showing how to create one static multimodal problem and evaluate a solution '''
from DDRB import DDRB
import numpy as np 

PID=3 # problem index (1,2,...,9,or 10), see DDRB class
DynaScn=0  # dynamic scenario (0,1,2,3,4,5,or 6), see DDRB class. DyaScn=0 means the problem is static

# The next command creates a problem object that stores both static and dynamic 
# attributes. Since the problem is static, the dynamic attributes will not be used  
problem=DDRB(PID,DynaScn)  
''' Up to this stage, you can perform ad hoc changes to the problem attributes.
e.g., you can set problem.statAttr.h_GO=.85 '''

# The next method calculates the problem data (e.g. rotation matrixes, locations of global minima). 
problem.calc_problem_data()  
# now the object problem has all the information required to evaluate a solution. 
''' *** Do NOT make any change to the problem after the following line *** '''

# Evaluation of a solution
x1=np.random.rand(problem.statAttr.dim)*(problem.statData.upBound-problem.statData.lowBound)+problem.statData.lowBound # a random solution in the search space
y1=problem.func_eval(x1) # call to the objective function.  

x2=problem.statData.globMinima[0] # This is the first global minimum of the problem  
y2=problem.func_eval(x2) # call to the objective function. 
 
print('for random solution x1, f(x1) = ' + str(y1))
print('for global minimum solution x2, f(x2) = ' + str(y2))
print('The global minimum value of the static problem is ' + str(problem.statData.globMinVal))
print('you can see the problem properties by typing "print(problem)"')