data {
  int<lower=0> N;
  int<lower=1> M;
  vector[M] x0;
  real t[N];
  matrix[N,M] x_obs;
}
parameters {
  real<lower=-10,upper=10> gamma;
  real<lower=0, upper=1> sigma[2];
}
model {
  matrix[N,M] x_hat;
  x_hat <- ho(t, x0, gamma);
  for (n in 1:N)
    x_obs[n] ~ normal(x_hat[n], sigma);
}
