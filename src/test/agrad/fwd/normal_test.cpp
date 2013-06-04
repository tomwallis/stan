#include <stan/prob/distributions/univariate/continuous/normal.hpp>
#include <gtest/gtest.h>
#include <stan/agrad/fvar.hpp>
#include <vector>

TEST(test,normal_allfvar_y) {
  using stan::agrad::fvar;
  using stan::prob::normal_log;

  fvar<double> y(0.5, 1);
  fvar<double> mu(0, 0);
  fvar<double> sigma(1, 0);

  fvar<double> logp = normal_log<false>(y, mu, sigma);
  
  EXPECT_FLOAT_EQ(-1.043938533, logp.val());
  EXPECT_FLOAT_EQ(-0.5, logp.tangent());
}

TEST(test,normal_allfvar_mu) {
  using stan::agrad::fvar;
  using stan::prob::normal_log;

  fvar<double> y(0.5, 0);
  fvar<double> mu(0, 1);
  fvar<double> sigma(1, 0);

  fvar<double> logp = normal_log<false>(y, mu, sigma);
  
  EXPECT_FLOAT_EQ(-1.043938533, logp.val());
  EXPECT_FLOAT_EQ(0.5, logp.tangent());
}

TEST(test,normal_allfvar_sigma) {
  using stan::agrad::fvar;
  using stan::prob::normal_log;

  fvar<double> y(0.5, 0);
  fvar<double> mu(0, 0);
  fvar<double> sigma(1, 1);

  fvar<double> logp = normal_log<false>(y, mu, sigma);
  
  EXPECT_FLOAT_EQ(-1.043938533, logp.val());
  EXPECT_FLOAT_EQ(-0.75, logp.tangent());
}


TEST(test,normal_yfvar_y) {
  using stan::agrad::fvar;
  using stan::prob::normal_log;

  fvar<double> y(0.5, 1);
  double mu(0);
  double sigma(1);

  fvar<double> logp = normal_log<false>(y, mu, sigma);
  
  EXPECT_FLOAT_EQ(-1.043938533, logp.val());
  EXPECT_FLOAT_EQ(-0.5, logp.tangent());
}

TEST(test,normal_mufvar_mu) {
  using stan::agrad::fvar;
  using stan::prob::normal_log;

  double y(0.5);
  fvar<double> mu(0, 1);
  double sigma(1);

  fvar<double> logp = normal_log<false>(y, mu, sigma);
  
  EXPECT_FLOAT_EQ(-1.043938533, logp.val());
  EXPECT_FLOAT_EQ(0.5, logp.tangent());
}

TEST(test,normal_sigmafvar_sigma) {
  using stan::agrad::fvar;
  using stan::prob::normal_log;

  double y(0.5);
  double mu(0);
  fvar<double> sigma(1, 1);

  fvar<double> logp = normal_log<false>(y, mu, sigma);
  
  EXPECT_FLOAT_EQ(-1.043938533, logp.val());
  EXPECT_FLOAT_EQ(-0.75, logp.tangent());
}


TEST(test,normal_vectorfvar_y) {
  using stan::agrad::fvar;
  using stan::prob::normal_log;
  using std::vector;

  vector<fvar<double> > y_vec;
  y_vec.push_back(fvar<double>(0.5, 1));
  y_vec.push_back(fvar<double>(0.5, 0));
  y_vec.push_back(fvar<double>(0.5, 0));
  double mu(0);
  double sigma(1);

  fvar<double> logp = normal_log<false>(y_vec, mu, sigma);
  
  EXPECT_FLOAT_EQ(-1.0439385333*3, logp.val());
  EXPECT_FLOAT_EQ(-0.5, logp.tangent());
}

TEST(test,normal_vectorfvar_mu) {
  using stan::agrad::fvar;
  using stan::prob::normal_log;
  using std::vector;

  vector<double> y_vec;
  y_vec.push_back(0.5);
  y_vec.push_back(0.5);
  y_vec.push_back(0.5);

  vector<fvar<double> > mu_vec;
  mu_vec.push_back(fvar<double>(0, 0));
  mu_vec.push_back(fvar<double>(0, 1));
  mu_vec.push_back(fvar<double>(0, 0));

  double sigma(1);

  fvar<double> logp = normal_log<false>(y_vec, mu_vec, sigma);
  
  EXPECT_FLOAT_EQ(-1.043938533*3, logp.val());
  EXPECT_FLOAT_EQ(0.5, logp.tangent());
}

TEST(test,normal_vectorfvar_sigma) {
  using stan::agrad::fvar;
  using stan::prob::normal_log;
  using std::vector;

  vector<double> y_vec;
  y_vec.push_back(0.5);
  y_vec.push_back(0.5);
  y_vec.push_back(0.5);
  vector<fvar<double> > mu_vec;
  mu_vec.push_back(fvar<double>(0,0));
  mu_vec.push_back(fvar<double>(0,0));
  mu_vec.push_back(fvar<double>(0,0));
  vector<fvar<double> > sigma_vec;
  sigma_vec.push_back(fvar<double>(1, 0));
  sigma_vec.push_back(fvar<double>(1, 0));
  sigma_vec.push_back(fvar<double>(1, 1));

  fvar<double> logp = normal_log<false>(y_vec, mu_vec, sigma_vec);
  
  EXPECT_FLOAT_EQ(-1.043938533*3, logp.val());
  EXPECT_FLOAT_EQ(-0.75, logp.tangent());
}
