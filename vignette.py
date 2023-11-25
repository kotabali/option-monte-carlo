import numpy as np               
import matplotlib.pyplot as plt  

#Paramètres du modèle de Black & Sholes :
T = 2         #temps avant maturité (en années)
r = 0.05      #taux sans risque
q = 0.02      #taux de dividendes
S0 = 100      #prix au temps 0
sigma = 0.2   #volatilité

#Paramètres des simulations :
n = T*252     #nombre de points sur une courbe (1 point par jour d'ouverture du marché)
dt = T/n      #intervalle de temps entre deux points successifs
N = 5         #nombre de simulations

#%% On simule les courbes dans l'univers risque-neutre

M = np.zeros((N,n))
M[:,0] = S0

np.random.seed(1)

Z = np.random.normal(size=(N,n-1))

for j in range(1,n) :
  M[:,j] = M[:,j-1] * np.exp((r-q-0.5*sigma**2)*dt + sigma*np.sqrt(dt)*Z[:,j-1])
  
#%% On trace les courbes
  
X = np.linspace(0,T,num=n)

for i in range(N) :
  plt.plot(X,M[i,:])