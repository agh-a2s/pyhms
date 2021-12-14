from sympy import *
from math import pi

# --------------------------------------------------------------------------------------
# 3-parameter model reduced to 2-parameter one
# Saturation relay experiment (1-point estimation, i.e., 2 parameters can be determined)

# Heat exchanger (delayed) model: 
# G=((b0+b0D*exp(-tau0*s))*exp(-tau*s))/(s^3+a2*s^2+a1*s+a0+a0D*exp(-theta*s)); Transfer function

# Since k=(b0+b0D)/(a0+a0D) => b0 = k*(a0+a0D) - b0D, one can write:
# G=((k*(a0+a0D) - b0D + b0D*exp(-tau0*s))*exp(-tau*s))/(s^3+a2*s^2+a1*s+a0+a0D*exp(-theta*s));

# Parameters to be set: p=[b0D tau0] 
# --------------------------------------------------------------------------------------

# --- INITIALIZATION ---
a2, a1, a0, a0D, b0, b0D, tau, tau0, theta, w, k, A, real = symbols(
    'a2 a1 a0 a0D b0 b0D tau tau0 theta w k A real')

G_mod2 = (a0 + a0D * cos(theta*w) - a2 * w**2)**2 + (a0D * sin(theta*w) - a1 * w + w**3)**2

ReG_n1 = (b0D * cos(w * (tau + tau0)) + b0 * cos(tau*w)) * (a0 + a0D * cos(theta*w) - a2 * w**2)
ReG_n2 = (b0D * sin(w * (tau + tau0)) + b0 * sin(tau*w)) * (a0D * sin(theta*w) - a1 * w + w**3)
ReG = (ReG_n1 + ReG_n2) / G_mod2

ImG_n1 = (b0D * cos(w * (tau + tau0)) + b0 * cos(tau*w)) * (a0D * sin(theta*w) - a1 * w + w**3)
ImG_n2 = (b0D * sin(w * (tau + tau0)) + b0 * sin(tau*w)) * (a0 + a0D * cos(theta*w) - a2 * w**2)
ImG = (ImG_n1 - ImG_n2) / G_mod2

# We want: ImG * Ra = 0, ReG * Ra = -1

# VER 2
ReG = ReG.subs(b0, k * (a0 + a0D) - b0D) 
ImG = ImG.subs(b0, k * (a0 + a0D) - b0D)

# --- RELAY EXPERIMENT DATA ---
K = 0.0322 # Static gain estimation
B = 100 # Relay-output amplitude
A_ = 0.5551 # Saturated relay-input amplitude (B/k, k=180.1451=1.4*Ra)

# Process-output (non-saturated)amplitude, period, and angular freq. (=omega_osc)
A0 = 0.971 
Tu0 = 369.7 
wu0 = 2 * pi / Tu0

Ra = 2 * B / pi * (1/A_ * asin(A_/A) + sqrt(A**2 - A_**2) / A**2)
Ra0 = Ra.subs(A, A0)  # Saturation relay gain (real-valued)

# VER2
#f10=(Ra0*ReG+1).subs([(k,w), (K, wu0)])   #.subs([(k, w),(K, wu0)])
#f20=(Ra0*ImG).subs([(k,w), (K, wu0)])
f10 = (Ra0 * ReG + 1).subs(k, K).subs(w, wu0)
f20 = (Ra0 * ImG).subs(k, K).subs(w, wu0)

cw = 0.1 # Weight
f = f10**2 + cw * f20**2 # Cost fun (unconstrained)

# --- INITIAL SETTING ---
# Variables: p=[b0D tau0]
# Constants: PP=[tau a2 a1 a0 a0D theta] - Based on the previous estimations
p = (0.5 * (0.5003 + 99.49907), 1)
PP = (136.7, 1, 2.02937 * 10**4, 0.5003, 99.49907, 0.5933) # For instance...

f = f.subs([(tau, PP[0]), (a2, PP[1]),  (a1, PP[2]), (a0, PP[3]), (a0D, PP[4]), (theta, PP[5])])
f = f.subs(w, 2)
f_fun = lambdify([b0D, tau0], f)

bounds = [(-100, 100), (0, 100)]

def relay(x):
    if len(x) != 2:
        raise ValueError("Problem is 2-dimensional")
    return f_fun(x[0], x[1])
