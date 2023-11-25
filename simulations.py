#%% Déclaration des paramètres

#Paramètres du modèle de Black & Sholes :
r = 0.05      #taux sans risque
q = 0.02      #taux de dividendes
S0 = 100      #prix au temps 0
sigma = 0.2   #volatilité

#Paramètres des options
T = 2         #temps avant maturité (en années)
K = 100       #prix d'exercice
b = 150       #barrière

#Paramètres des simulations :
n = T*252     #nombre de points sur une courbe (1 point par jour d'ouverture du marché)
dt = T/n      #intervalle de temps entre deux points successifs
N = 100000    #nombre de simulations

#%% On simule les courbes dans l'univers risque-neutre sur CPU

import numpy as np  

def monte_carlo_cpu() :             
  #On génère les gaussiennes iid
  np.random.seed(1)
  Z = np.random.normal(size=(N,n-1))

  #On simule les N trajectoires
  M = np.zeros((N,n))
  M[:,0] = S0
  for j in range(1,n) :
    M[:,j] = M[:,j-1] * np.exp((r-q-0.5*sigma**2)*dt + sigma*np.sqrt(dt)*Z[:,j-1])

  return M

M_cpu = monte_carlo_cpu()

#%% On simule les courbes dans l'univers risque-neutre sur GPU

import cupy as cp  

def monte_carlo_gpu() :             
  #On génère les gaussiennes iid
  cp.random.seed(1)
  Z = cp.random.normal(size=(N,n-1))

  #On simule les N trajectoires
  M = cp.zeros((N,n))
  M[:,0] = S0
  for j in range(1,n) :
    M[:,j] = M[:,j-1] * cp.exp((r-q-0.5*sigma**2)*dt + sigma*cp.sqrt(dt)*Z[:,j-1])
    
  return M

M_gpu = monte_carlo_gpu()

#%% Comparaison des performances

%timeit monte_carlo_cpu()
%timeit monte_carlo_gpu()

#%% Options asiatiques

def asiatique_cpu(M) :
  gain = np.maximum(M.mean(axis=1)-K,0)
  return float(gain.mean())*np.exp(-r*T)

def asiatique_gpu(M) :
  gain = cp.maximum(M.mean(axis=1)-K,0)
  return float(gain.mean())*cp.exp(-r*T)

print(asiatique_cpu(M_cpu),asiatique_gpu(M_gpu))

#%% Options barrières

def barriere_cpu(M) :
  gain = np.any(M > b,axis=1)*np.maximum(M[:,-1]-K,0)
  return float(gain.mean())*np.exp(-r*T)

def barriere_gpu(M) :
  gain = cp.any(M > b,axis=1)*cp.maximum(M[:,-1]-K,0)
  return float(gain.mean())*cp.exp(-r*T)

print(barriere_cpu(M_cpu),barriere_gpu(M_gpu))

#%% Options rétrospectives

def retrospective_cpu(M) :
  gain = np.maximum(np.max(M,axis=1)-K,0)
  return float(gain.mean())*np.exp(-r*T)

def retrospective_gpu(M) :
  gain = cp.maximum(cp.max(M,axis=1)-K,0)
  return float(gain.mean())*cp.exp(-r*T)

print(retrospective_cpu(M_cpu),retrospective_gpu(M_gpu))

#%% Temps d'exécution

%timeit asiatique_cpu(M_cpu)
%timeit asiatique_gpu(M_gpu)

%timeit barriere_cpu(M_cpu)
%timeit barriere_gpu(M_gpu)

%timeit retrospective_cpu(M_cpu)
%timeit retrospective_gpu(M_gpu)
