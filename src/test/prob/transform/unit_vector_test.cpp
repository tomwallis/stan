#include <vector>
#include <cmath>
#include <stdexcept>
#include <gtest/gtest.h>

#include <stan/agrad/agrad.hpp>
#include <stan/prob/transform/unit_vector_constrain.hpp>
#include <stan/prob/transform/unit_vector_free.hpp>
#include <stan/math/matrix/determinant.hpp>

using Eigen::Matrix;
using Eigen::Dynamic;

TEST(prob_transform,unit_vector_rt0) {
  Matrix<double,Dynamic,1> x(4);
  x << 0.0, 0.0, 0.0, 0.0;
  Matrix<double,Dynamic,1> y = stan::prob::unit_vector_constrain(x);
  EXPECT_NEAR(0, y(0), 1e-8);
  EXPECT_NEAR(0, y(1), 1e-8);
  EXPECT_NEAR(0, y(2), 1e-8);
  EXPECT_NEAR(0, y(3), 1e-8);
  EXPECT_NEAR(1.0, y(4), 1e-8);

  Matrix<double,Dynamic,1> xrt = stan::prob::unit_vector_free(y);
  EXPECT_EQ(x.size()+1,y.size());
  EXPECT_EQ(x.size(),xrt.size());
  for (Matrix<double,Dynamic,1>::size_type i = 0; i < x.size(); ++i) {
    EXPECT_NEAR(x[i],xrt[i],1E-10);
  }
}
TEST(prob_transform,unit_vector_rt) {
  Matrix<double,Dynamic,1> x(3);
  x << 1.0, -1.0, 1.0;
  Matrix<double,Dynamic,1> y = stan::prob::unit_vector_constrain(x);
  Matrix<double,Dynamic,1> xrt = stan::prob::unit_vector_free(y);
  EXPECT_EQ(x.size()+1,y.size());
  EXPECT_EQ(x.size(),xrt.size());
  for (Matrix<double,Dynamic,1>::size_type i = 0; i < x.size(); ++i) {
    EXPECT_FLOAT_EQ(x[i],xrt[i]) << "error in component " << i;
  }
}
TEST(prob_transform,unit_vector_match) {
  Matrix<double,Dynamic,1> x(3);
  x << 1.0, -1.0, 2.0;
  double lp;
  Matrix<double,Dynamic,1> y = stan::prob::unit_vector_constrain(x);
  Matrix<double,Dynamic,1> y2 = stan::prob::unit_vector_constrain(x,lp);

  EXPECT_EQ(4,y.size());
  EXPECT_EQ(4,y2.size());
  for (Matrix<double,Dynamic,1>::size_type i = 0; i < x.size(); ++i)
    EXPECT_FLOAT_EQ(y[i],y2[i]) << "error in component " << i;
}
TEST(prob_transform,unit_vector_f_exception) {
  Matrix<double,Dynamic,1> y(2);
  y << 0.5, 0.55;
  EXPECT_THROW(stan::prob::unit_vector_free(y), std::domain_error);
  y << 1.1, -0.1;
  EXPECT_THROW(stan::prob::unit_vector_free(y), std::domain_error);
}
TEST(probTransform,unit_vector_jacobian) {
  using stan::agrad::var;
  using std::vector;
  var a = 2.0;
  var b = 3.0;
  var c = -1.0;
  
  Matrix<var,Dynamic,1> y(3);
  y << a, b, c;
  
  var lp(0);
  Matrix<var,Dynamic,1> x 
    = stan::prob::unit_vector_constrain(y,lp);
  
  vector<var> indeps;
  indeps.push_back(a);
  indeps.push_back(b);
  indeps.push_back(c);

  vector<var> deps;
  deps.push_back(x(0));
  deps.push_back(x(1));
  deps.push_back(x(2));
  deps.push_back(x(3));
  
  vector<vector<double> > jacobian;
  stan::agrad::jacobian(deps,indeps,jacobian);

  Matrix<double,Dynamic,Dynamic> J(4,4);
  for (int m = 0; m < 4; ++m) {
    for (int n = 0; n < 3; ++n) {
      J(m,n) = jacobian[m][n];
    }
    J(m,3) = x(m).val(); 
  }
  
  double det_J = J.determinant();
  double log_det_J = log(fabs(det_J));

  EXPECT_FLOAT_EQ(log_det_J, lp.val()) << "J = " << J << std::endl << "det_J = " << det_J;
  
}
