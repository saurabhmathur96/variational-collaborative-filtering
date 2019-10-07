'''pmf.py: probabilistic matrix factorization using variational inference

Model: 
U: user vectors of rank n, Ixn
V: item vectors of rank n, Jxn
M: Ratings matrix, IxJ

m_ij = rating for item j given by user i.


Likelihood: 
P(m_ij | u_i, v_j) = N((u_i)^T v_j, tau^2)

Priors: 
P(u_i) = prod_l=1^n N(u_il, diag(nu_il^2))
P(v_j) = prod_l=1^n N(v_jl, diag(rho_i^2))

Posterior:
P(U,V|M) = P(M|U,V)P(U)P(V)/P(M)

=> Intractable due to P(M)

Maximum a Posteriori:
U, V = argmax(U, V) P(U,V|M) 

(MaP gets rid of P(M), so it is doable)

Approximate posterior estimation (This project):
Exact inference is intractable
=> So, perform variational inference

Let Q(U, V) be the approximate posterior distribution.
Applying the mean-field approximation, Q is factorized as:
Q(U, V) = Q(U) Q(V)

Let Q(U) and Q(V) be gaussians such that:
Q(u_i) = N(u_i, phi_i)
V(v_j) = N(v_j, psi_j)

The code below maximizes the variational lower bound F(Q(U), Q(V)),
with respect to variational parameters U, Phi, V and Psi (E-step) and
model parameters sigma^2, rho^2 and tau^2.
'''
import numpy as np
from tqdm.auto import trange

def user_params(user_count, rank):
  u = np.random.normal(loc=0, scale=1, size=(user_count, rank))

  phi = np.zeros((user_count, rank, rank))

  return (u, phi)

def item_params(item_count, rank):
  v = np.random.normal(loc=0, scale=1, size=(item_count, rank))

  psi = np.zeros((item_count, rank, rank))

  return (v, psi)

def model_params(user_count, item_count, rank):
  tau2 = 1

  u0 = np.zeros(rank)
  sigma2 = np.ones(rank)
  v0 = np.zeros(rank)
  rho2 = np.ones(rank) * 1/rank   

  return (tau2, sigma2, rho2)

def var_inference(ratings, tau2, sigma2, rho2, u, phi, v, psi):
  user_count = len(u)
  item_count = len(v)
  rank = len(sigma2)

  S = np.array([np.diag(rho2) for _ in range(item_count)])
  t = np.zeros((item_count, rank))

  inv = np.linalg.inv
  for user in range(user_count):
    sum_term = sum(psi[j] + np.outer(v[j], v[j]) 
                   for i, j, r in ratings if i == user)
    sum_term = sum_term/tau2 + np.zeros((rank, rank))
    phi_ = inv(np.diag(1/sigma2) + sum_term)
    
    phi[user] = phi_
    sum_term = sum(r * v[j]
                   for i, j, r in ratings if i == user)
    sum_term = sum_term/tau2 + np.zeros(rank)
    u[user] = np.dot(phi_, sum_term)

    for item, r in ((j, r) for i, j, r in ratings if i == user):
      S[item] += (phi_ + np.outer(u[user], u[user])) / tau2
      t[item] += (r * u[user]) / tau2

  for item in range(item_count):
    psi[item] = inv(S[item])
    v[item] = np.dot(psi[item], t[item])

  return ((u, phi), (v, psi))

def expectation(ratings, tau2, sigma2, rho2, u, phi, v, psi):
  
  return var_inference(ratings, tau2, sigma2, rho2, u, phi, v, psi)

def maximization(ratings, tau2, sigma2, rho2, u, phi, v, psi):
  user_count = len(u)
  item_count = len(v)
  rank = len(sigma2)

  for l in range(rank):
    sum_term = sum(phi[j, l, l] + u[j, l]**2 for j in range(user_count))
    sigma2[l] = sum_term/(user_count-1)
  
   
  sum_term = 0.0
  for i, j, r in ratings:
    part1 = phi[i] + np.outer(u[i], u[i])
    part2 = psi[j] + np.outer(v[j], v[j])
    sum_term += r**2 - 2*r*np.dot(u[i], v[j]) + np.sum(part1 * part2) # tr(AB)
  
  tau2 = sum_term/(len(ratings)-1)

  return tau2, sigma2, rho2

def error(ratings, u, v):
    return np.sqrt(np.mean([( np.dot(u[i], v[j]) - m)**2 for i, j, m in ratings]))