library(rstan)
set.seed(0)

## assumed time is first column
dat <- read.csv('ho.csv')

## 
N <- nrow(dat)-1
M <- ncol(dat)-1

## initial condition
t0 <- dat[1,1]; 
x0 <- unlist(dat[1,1+(1:M)])

## observations
sigma <- 0.15
t <- dat[1+(1:N), 1]
x <- dat[1+(1:N), 1+(1:M)]
noise <- matrix(rnorm(N*M, 0, sigma), N, M);
x_obs <- x + noise

stan_rdump(c('N', 'M', 't0', 'x0', 't', 'x_obs'),
           file='ho.sim.data.R')
