''' An example showing how to create one dynamic multimodal problem and evaluate
a solution at different times'''

from DDRB import DDRB
import numpy as np 

PID=2 # problem index (1,2,...,9,or 10), see DDRB class
DynaScn=5  # dynamic scenario (0,1,2,3,4,5,or 6), see DDRB class.  

# The next command creates a problem object that stores both static and dynamic 
# attributes.  
problem=DDRB(PID,DynaScn)  
''' Up to this stage, you can perform ad hoc changes to the problem attributes.
e.g., you can set problem.statAttr.h_GO=.85 '''

# The next method calculates the problem data (e.g. rotation matrixes, locations of global minima). 
problem.calc_problem_data()  
# now the object problem has all the information required to evaluate a solution. 
''' *** Do NOT make any change to the problem after the following line *** '''

# Evaluation of a solution at different times
x1=np.random.rand(problem.statAttr.dim)*(problem.statData.upBound-problem.statData.lowBound)+problem.statData.lowBound # a random solution in the search space
print('for random solution x1:')

problem.numCallObj=int(1e5) # for testing purpose, you can change this property to observe its effect on the solution value.
y1=problem.func_eval(x1) # call to the objective function when 1,000,000 evalautions has been used so far.  
print('\tf(x1) = ' + str(y1) + ' if ' + str(problem.numCallObj-1) + ' evalautions has alreeady been used')

problem.numCallObj=int(2e5) # for testing purpose, you can change this property to observe its effect on the solution value.
y2=problem.func_eval(x1) # # call to the objective function when 2,000,000 evalautions has been used so far.  
print('\tf(x1) = ' + str(y2) + ' if ' + str(problem.numCallObj-1) + ' evalautions has alreeady been used')

problem.numCallObj=int(3e5) # for testing purpose, you can change this property to observe its effect on the solution value.
y3=problem.func_eval(x1) # # call to the objective function when 2,000,000 evalautions has been used so far.  
print('\tf(x1) = ' + str(y3) + ' if ' + str(problem.numCallObj-1) + ' evalautions has alreeady been used')

print('\nyou can see the problem properties by typing "print(problem)"')