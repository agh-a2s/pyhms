# SOFTX-D-21-00154
PyDDRBG: A Python framework for benchmarking and evaluating static and dynamic multimodal optimization methods


This is the Python code for dynamic distortion and rotation benchmark (DDRB) generator [1]. Please refer to the corresponding publications for further details on this benchmark generator for dynamic multimodal optimization. The software details have been provided in the corresponding publication in “SoftwareX”. 
There are two simple and one comprehensive example to show how to use this code:
“Example_static.py” is an example showing how to create a static multimodal problem and evaluate a solution using PyDDRBG
“Example_dynamic.py” is an example showing how to create a dynamic multimodal problem and evaluate a solution using PyDDRBG
“Example_optim.py” is a more comprehensive example showing how to generate a customized dynamic problem, optimize it using a simple but powerful optimization method, and calculate the performance based on the robust peak ratio indicator.   

This code has been verified with Python 3.7.9. For feedback on this code, please contact Ali Ahrari at aliahrari1983@gmail.com. 

Reference
[1] Ahrari, Ali, Saber Elsayed, Ruhul Sarker, Daryl Essam, and Carlos A. Coello Coello. "A Novel Parametric Benchmark Generator for Dynamic Multimodal Optimization." Swarm and Evolutionary Computation (2021): DOI: j.swevo.2021.100924


