# =============================================================================
# Basic functions from which the composite functions can be generated

# Written by Ali Ahrari (aliahrari1983@gmail.com)
# last update by Ali Ahrari on 15 March 2021
# =============================================================================

class BasicFunc:

    @staticmethod
    def bohach2(z,h_GO): # f1: Bohachevsky 2 function (Modified)
        A=0.5 # by default, it is 0.3. A larger value means a harder problem
        x1=z[0::2]
        x2=z[1::2]
        f= x1**2+2*x2**2-A*h_GO*np.cos(3*np.pi*x1)*np.cos(4*np.pi*x2)+A*h_GO
        return f
        
    def booth(x): # f2: Booth function - shifted 2D function, see https://www.sfu.ca/~ssurjano/booth.html
        x1=x[0]+1
        x2=x[1]+3
        f = ( (x1+2*x2-7)**2 + (2*x1+x2-5)**2)
        return f
         
    def branin(z):   # f3: multi-output rescaled Branin function, 
        x=z[0::2]*4+3
        y=z[1::2]*4+7
        px=  (x<-5)*(x+5)**2   +    (x>10)*(x-10)**2
        py=  (y<0)*y**2        +    (y>15)* (y-15)**2
        f=(y-5.1/(4*np.pi**2)*x**2+5/np.pi*x-6)**2 + 10*(1-.125/np.pi)*np.cos(x)+10
        f=f+ px+py
        f=np.sqrt(f)-np.sqrt(5/(4*np.pi))
        f=np.clip(f,a_min=0,a_max=np.inf)
        return f
     
    def cmmp(x): # f4: x1*=+- sqrt(  27/7 )  x2= +- sqrt(4/7), f*=31/7
    # see "Multimodal Optimization Using a Bi-Objective Evolutionary 
    # Algorithm" by Deb and Saha for the structure
        f0=31/7 
        x1=x[0::2]
        x2=x[1::2]
        g1=(x1/2)**2+(x2/4)**2-1
        g2=(x1/3)**2+(x2/1)**2-1
        g1=np.abs(g1)*(g1<0)
        g2=np.abs(g2)*(g2<0)
        f=x1**2+x2**2+ 1e10*((g1+g2)+((g1+g2)>0))-f0
        f= np.clip(f,a_min=0,a_max=np.inf) # correction for possible rounding error
        return f  
        
    def dp(x): # f5: Dixon & Price function (Shifted): single-output 
        n=np.arange(2,x.size+1)
        x=x+2.0**(2.0**(1-np.arange(1,x.size+1)))/2
        f=(x[0]-1)**2+sum(n*(2*x[1:]**2-x[0:-1])**2)
        return f
        
    def griewank(x,h_GO): # f6: multi-output rescaled 2D Griewank function 
        x=x*10
        x1=x[0::2]
        x2=x[1::2]
        f=  (x1**2+x2**2)/100 - h_GO*np.cos(x1/1)*np.cos(x2/np.sqrt(2)) + h_GO
        f=1000*f
        return f
    
    def himl(z): # f7: Himmelblau function (Rescaled)
        z=3*z
        x=z[0::2]
        y=z[1::2]
        f=(x**2+y-11)**2 + (x+y**2-7)**2
        return f
        
        
    def hump6(z): # f11: Six Hump Camel Back function (Rescaled): nonlinearly rescaled 6-hump function
        a=0.1 # specifies the distortion (a=0 for default function)
        x=z[0::2]
        y=z[1::2]
        x=x*(1+a*np.sign(x))
        y=y*(1-a*np.sign(y))
        f=(4-2.1*x**2+x**4/3)*x**2   +    x*y     +   (-4+4*y**2)*y**2  +1.03162845349
        return f
        
    def hump3(z,h_GO): # f13: multi-output three-Hump Camel Function (Modified)
        x=z[0::2]
        y=z[1::2]
        Q2= (x**2+x**6+y**2)*.02
        Q1=(2*x**2-1.05*x**4+x**6/6+x*y+y**2)
        f=100*(h_GO*Q1+(1-h_GO)*Q2)
        return f
        
    def lvn13(z,h_GO): # f8: multi-output shifted 2d Levy function N13 
                       # see https://www.sfu.ca/~ssurjano/levy13.html for original function
        x1=z[0::2]*2+1
        x2=z[1::2]*2+1
        f=h_GO*(np.sin(3*np.pi*x1))**2 + (x1-1)**2*(1+h_GO*(np.sin(3*np.pi*x2))**2) + (x2-1)**2*(1+h_GO*(np.sin(2*np.pi*x2))**2)
        return f

    def neumaier3(x): # f9: neumaier3 aka Trid function
        D=x.size
        ind=np.arange(1,D+1)
        sh=ind*(D+1-ind)
        fstar=-D*(D+4)*(D-1)/6
        x=x+sh
        f=np.sum((x-1)**2) - np.sum(x[0:-1]*x[1:]) - fstar    
        return f

    def shubert(x): # f10: Shubert with penalty (Rescaled)
        x=4*x
        f = 1
        k = np.arange(1,6)
        D=x.size
        for m in np.arange(D):
            f=f*np.sum(  k*np.cos(  (k+1)*x[m]+k  )  )
        p=np.abs(x)-10
        penalt= p**2*(p>0)
        offset=np.array([12.8708854977257, 1.86730908831024e+02, 2.70909350557283e+03 ])
        if D>=4:
            sys.exit('The rescaled shubert function is not defined for this dimension Available dimensions are 1,2,3')
        f=f+offset[D-1]
        f=f/10**(D-1)
        f=f+sum(penalt)
        return f
   
    def weierstrass(y,h_GO):  # f14: Weierstrass function (Modified)       
        D=y.size
        y=.1*y
        a=np.sqrt(h_GO)*0.5 # controls the global basin (a higher a makes problem harder)
        b=3 # a higher value makes it makes it more rugged-default is 3
        k=np.arange(4) # level of optima
        h=np.zeros(D)
        for m in np.arange(D):
            h[m]=np.sum(a**k*np.cos(2*np.pi*b**k*(y[m]+.5)))
        hprime=sum(a**k*np.cos(np.pi*b**k))
        P= np.sum(  (np.abs(y)-.5)**1 *(np.abs(y)>.5)   )
        f=50*( sum(h-hprime) + P )
        return f

    def zakharov(x): # f15: zakharov function (Modified)
        n=np.arange(1,x.size+1)
        g=(0.5*sum(n*x))**2
        f=np.sqrt(np.sum(x**2)+g*(1+g))
        return f

import numpy as np 
import sys  

if __name__=='__main__':
    x=np.array([1,2,3])/3.28
    h_GO=.5
    y=BasicFunc.zakharov(x)
    print(y)
     