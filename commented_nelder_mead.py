# min = -310.193 argmin = (-4.25,0)
import numpy as np 
import random 

import matplotlib.pyplot as plt
from matplotlib import collections  as mc
plt.style.use('dark_background')

np.random.seed(42)

# function used by the algorithm
def f(x):
    return 18 + 11.4*x[0] - 31*(x[0]**2) + 0.6*(x[0]**3) + x[0]**4 + 50*(x[1]**2) + random.gauss(0,15)

# function used to plot
def f2(x,y):
    return 18 + 11.4*x - 31*(x**2) + 0.6*(x**3) + x**4 + 50*(y**2) 

def set_plot_parameters():
    levels = np.arange(-400,1400,100)
    x = np.linspace(-7.0, 7.0,1000)
    y = np.linspace(-7.0, 7.0,1000)
    X, Y = np.meshgrid(x, y)
    Z = f2(X,Y)
    return X, Y, Z, levels

def contour_plot(simplex,iters,score,X,Y,Z,levels):
    fig, ax = plt.subplots(figsize=(12,7))
    CS = plt.contour(X, Y, Z, levels=levels, cmap='autumn')
    ax.clabel(CS, inline=1, fontsize=10)
    ax.set_title('Iteration '+str(iters)+": "+str(round(score,3)))
    
    lines = [ [simplex[0],simplex[1]], [simplex[1],simplex[2]], 
            [simplex[2],simplex[0]] ]
    lc = mc.LineCollection(lines, colors='white', linewidths=1)
    ax.add_collection(lc)
    ax.autoscale()
    ax.margins(0.1)

    plt.pause(1/(iters*10)**2)
    plt.close()

def update(simplex,res,point,score):
    # update the simplex
    simplex[2] = point
    # substitute the new tuple (x,f(x)) in the score list
    res[2] = point, score
    return simplex, res

def contract(simplex,res,contraction,worst_score):
    contraction_score = f(contraction)
    if contraction_score <= worst_score:
        simplex, res = update(simplex,res,contraction,contraction_score)
    else:
        # perform shrinkrage
        simplex = shrink(simplex,0.9) # 0.5 va male
        # substitute the new tuples (x,f(x)) in the score list
        res[1] = simplex[1],f(simplex[1])
        res[2] = simplex[2],f(simplex[2])
    return simplex, res

def shrink(simplex,sigma=0.5):
    for i in [ 1, 2 ]:
        simplex[i] = simplex[0] + sigma * ( simplex[i] - simplex[0] )
    return simplex

def nelder_mead(f,start,max_iter=5000,max_iter_no_change=5,toll=0.001,plot=True,display=True):
    if display:
        print('\nNelder-Mead Algorithm Started')
    simplex = start
    iters = 0
    iter_no_change = 0
    best_score_list = []

    # generate tuples (X,f(X)) and sort the score list and the simplex on the scores
    res = [ (x,f(x)) for x in simplex ]
    res.sort(key=lambda w: w[1])
    simplex = [ x for x,fx in res ]
    
    # in order to not recompute meshgrids at each iteration
    X, Y, Z, levels = set_plot_parameters()
    
    while iters < max_iter and iter_no_change < max_iter_no_change:
        iters += 1

        # retrieve ordered tuples (X,f(X))
        best, best_score = res[0]
        second_worst, second_score = res[1]
        worst, worst_score = res[2]

        best_score_list.append(best_score)

        # check if best_score has changed from the previous iteration more than toll
        if iters > 1:
            if np.abs( best_score_list[iters-1] - best_score_list[iters-2] ) <= toll:
                iter_no_change += 1
            else:
                iter_no_change = 0 

        # compute the centroid
        centroid = ( best + second_worst ) / 2

        # compute and evaluate the reflective point
        reflection = centroid - ( worst - centroid )
        reflection_score = f(reflection)
        
        if reflection_score <= best_score:
            # compute and evaluate the expanded point
            expansion = 2 * reflection - centroid
            expansion_score = f(expansion)
            # expansion 
            if expansion_score <= reflection_score:
                simplex, res = update(simplex,res,expansion,expansion_score)
            # reflection
            else:
                simplex, res = update(simplex,res,reflection,reflection_score)
        
        # reflection
        elif best_score < reflection_score <= second_score:
            simplex, res = update(simplex,res,reflection,reflection_score)
        
        # contraction 1 or shrink
        elif second_score < reflection_score <= worst_score:
            # compute and evaluate the contracted point
            contraction = centroid + ( reflection - centroid ) / 2
            simplex, res = contract(simplex,res,contraction,worst_score)

        # contraction 2 or shrink   
        else:
            # compute and evaluate the contracted point 
            contraction = centroid + ( worst - centroid ) / 2
            simplex, res = contract(simplex,res,contraction,worst_score)
            
        # sort the list (x,f(x)) and the simplex on the scores
        res.sort(key=lambda w: w[1])
        simplex = [ x for x,fx in res ]
        
        if plot:
            contour_plot(simplex, iters, res[0][1], X, Y, Z, levels)
            plt.show
    
    if display:
        if iters == max_iter:
            print(' - Maximum number of iterations reached:',iters)
        else:
            print(' - Maximum number of iter_no_change reached:',iter_no_change)
            print(' - Minimum reached at iteration',iters-iter_no_change)
        print('\nx_min:',res[0][0],'\nscore:',res[0][1])
    return res[0]

# find a suitable simplex by taking the 3 minima over several repetitions
# of the algorithm with random starting points
def find_simplex(x,y,std,iterations=50):
    scores = []
    argmins = []
    for i in range(iterations):
        simplex = []
        for i in range(3):
            # generate two random numbers with mean x and y and standard deviation std
            simplex.append(np.array([random.gauss(x,std),random.gauss(y,std)]))

        argmin, score = nelder_mead(f,simplex,toll=1,plot=False,display=False)
        scores.append(score)
        argmins.append(argmin)

    new = [ (argmin, score) for argmin, score in zip(argmins,scores) ]
    new.sort(key=lambda w: w[1] )

    return [new[0][0], new[1][0], new[2][0] ]

# find the exact argmin once a suitable starting simplex (sss) has been found
# e.g. get rid of noise running multiple times the algorithm from the sss
def estimate(n_iterations=50,plot=True):
    x_list = []
    for i in range(n_iterations):
        print('Iteration',i,end='\r')
        simplex = find_simplex(0,5,20)
        x_min, _ = nelder_mead(f,simplex,toll=1,plot=False,display=False)
        x_list.append(x_min)
    x_array = np.array(x_list)
    centroid = (x_array[:,0].mean(),x_array[:,1].mean())

    if plot:
        X, Y, Z, levels = set_plot_parameters()
        fig, ax = plt.subplots(figsize=(12,7))
        CS = plt.contour(X, Y, Z, levels=levels, cmap='autumn')
        ax.clabel(CS, inline=1, fontsize=8)
        ax.set_title("Minimum at: "+str(centroid))
        plt.scatter(x_array[:,0],x_array[:,1],c='green',s=5,marker='.')
        plt.scatter(centroid[0],centroid[1],c='white',s=20)
        plt.show()
    
    print("Average minimum:",centroid)
        
#x_min, score = nelder_mead(f,[np.array([0,0]),np.array([-5,4]),np.array([-5,-4])],toll=1)
estimate()

# reference http://www.scholarpedia.org/article/Nelder-Mead_algorithm

# Nelder-Mead base:
# 1) due vertici vicini ai minimi e il simplesso degenera 
#    np.array([-4.5,0.5]),np.array([4,0]),np.array([2,3]) --- NON CON SHRINKRAGE
# 2) converge sul minimo non assoluto se i punti iniziali sono troppo a destra 
#    del centro o se un singolo punto è molto vicino al minimo locale
#    np.array([4,0]),np.array([-5,4]),np.array([-5,-4])
# 3) si pianta nella sella fra i minimi 
#    np.array([1,1]),np.array([-1,-3]),np.array([3,3])
# 4) simplessi iniziali troppo schiacciati performano di merda
# buono visivamente np.array([0,4]),np.array([-2,5]),np.array([0,6])