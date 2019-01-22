#include <RcppArmadillo.h>
#include <omp.h>
#include <trng/discrete_dist.hpp>
#include <trng/yarn2.hpp>
using namespace arma;

#pragma omp declare reduction(+                    \
: arma::mat                                        \
  : omp_out += omp_in)                             \
    initializer(omp_priv = omp_orig)
    
#pragma omp declare reduction(+                    \
    : arma::vec                                    \
      : omp_out += omp_in)                         \
        initializer(omp_priv = omp_orig)
        
        // [[Rcpp::depends(RcppArmadillo)]]
        // [[Rcpp::plugins(openmp)]]
        
        arma::vec sumk_img(arma::cube &img, int k, arma::mat &C)
        {
          arma::vec sum = {0, 0, 0};
          uword i, j;
          
#pragma omp parallel for schedule(static)       \
          reduction(+                           \
            : sum) private(i, j)                \
            shared(img, C, k) collapse(2)
            for (i = 0; i < C.n_rows; i++)
            {
              for (j = 0; j < C.n_cols; j++)
              {
                if (C(i, j) == k)
                  sum += img.tube(i, j);
              }
            }
            return sum;
        }
      
      arma::mat S_k(arma::cube &img, int k, arma::mat C, arma::vec &mu)
      {
        mat S_k = zeros<mat>(3, 3);
        uword i, j;
#pragma omp parallel for schedule(static)     \
        reduction(+                           \
          : S_k) private(i, j)                \
          shared(img, k, C, mu) collapse(2)
          for (i = 0; i < C.n_rows; i++)
          {
            for (j = 0; j < C.n_cols; j++)
            {
              arma::vec p_ij = img.tube(i, j);
              arma::vec d = p_ij - mu;
              if (C(i, j) == k)
                S_k += d * d.t();
            }
          }
          return S_k;
      }
      
      inline int mirrorIndex(int fetchI, int length)
      {
        return (fetchI < 0) + (fetchI >= length) * (length - 1);
      }
      
      arma::mat log_Normal(arma::mat &C, arma::mat &img_vec,
                           double alpha, double beta, arma::mat &mu, arma::cube &Sigma)
      {
        vec K1 = unique(C);
        int K = K1.n_elem;
        int m = C.n_rows;
        int n = C.n_cols;
        mat Pro(K, m * n);
        int k;
        for (k = 1; k <= K; k++)
        {
          mat X = img_vec.each_col() - mu.col(k - 1);
          mat L = chol(Sigma.slice(k - 1), "lower");
          //img_vec = inv(L) * img_vec;
          mat Y = solve(L, X);
          mat Z = -sum(Y % Y) / 2;
          Z -= accu(log(L.diag()));
          Pro.row(k - 1) = Z;
        }
        return Pro;
      }
      
      // [[Rcpp::export]]
      arma::mat Segment(arma::cube &img, arma::mat &C, arma::mat &img_vec, int GibbsN, double alpha, double beta,
                        arma::vec &mu_0, arma::mat &Lambda_0, double v_0, arma::mat &Sigma_0)
      {
        vec K1 = unique(C);
        int K = K1.n_elem;
        vec nk(K);
        
        int m = img.n_rows;
        int n = img.n_cols;
        
        //每一步迭代的mu，Sigma，k=1,2,...,K
        mat mu = randu<mat>(3, K);
        cube Sigma(3, 3, K);
        for (int i = 0; i < K; i++)
        {
          Sigma.slice(i) = eye<mat>(3, 3);
        }
        mat postC(m, n);
        
        int it;
        for (it = 0; it < GibbsN; it++)
        { //-----------------------------------------------------------------------
          
          for (int i = 1; i <= K; i++)
          {
            uvec cnt = find(C == i);
            nk(i - 1) = cnt.n_elem;
          }
          
          //update mu_k,Sigma_k
          int k;
          for (k = 1; k <= K; k++)
          {
            //update mu_k
            mat Sigma_kinv = Sigma.slice(k - 1).i();
            mat Sigma_quta_inv = nk(k - 1) * Sigma_kinv + Lambda_0.i();
            vec rhs = Sigma_kinv * sumk_img(img, k, C) + Lambda_0.i() * mu_0;
            vec mu_quta = solve(Sigma_quta_inv, rhs);
            //posterior of mu
            mu.col(k - 1) = mvnrnd(mu_quta, Sigma_quta_inv.i());
            
            //update Sigma_k
            vec muk = mu.col(k - 1);
            mat Wis_S_inv = Sigma_0.i() + S_k(img, k, C, muk);
            int Wis_df = v_0 + nk(k - 1);
            mat S = iwishrnd(Wis_S_inv, Wis_df);
            Sigma.slice(k - 1) = S;
          }
          
          //update C_ij
          
          mat logN = log_Normal(C, img_vec, alpha, beta, mu, Sigma);
          
#pragma omp parallel
{
  trng::yarn2 rx;
  rx.seed(10);
  int size = omp_get_num_threads(); // get total number of processes
  int rank = omp_get_thread_num();  // get rank of current process
  rx.split(size, rank);
  vec Pro(K);
  vec Pot_f(K);
  trng::discrete_dist distSampling(Pro.begin(), Pro.end());
  uword i, j;
  
#pragma omp parallel for schedule(static) private(i, j) \
  collapse(2)
    for (i = 0; i < m; i++)
    {
      for (j = 0; j < n; j++)
      {
        if (i % 2 == j % 2)
        {
          for (k = 1; k < K; k++)
          {
            double tem;
            tem += (C.at(mirrorIndex(i - 1, m), j) == k ? alpha : beta);
            tem += (C.at(mirrorIndex(i + 1, m), j) == k ? alpha : beta);
            tem += (C.at(i, mirrorIndex(j + 1, n)) == k ? alpha : beta);
            tem += (C.at(i, mirrorIndex(j - 1, n)) == k ? alpha : beta);
            
            Pot_f(k - 1) = 2 * tem;
          }
          Pro = Pot_f + logN.col(m * j + i);
          Pro = Pro - Pro.max();
          //cout <<Pro<<endl;
          Pro = exp(Pro);
          Pro = Pro / sum(Pro);
          distSampling = trng::discrete_dist(Pro.begin(), Pro.end());
          postC(i, j) = distSampling(rx);
        }
      }
    }
    
#pragma omp parallel for schedule(static) private(i, j) \
    collapse(2)
      for (i = 0; i < m; i++)
      {
        for (j = 0; j < n; j++)
        {
          if ((i + 1) % 2 == j % 2)
          {
            for (k = 1; k < K; k++)
            {
              double tem;
              tem += (C.at(mirrorIndex(i - 1, m), j) == k ? alpha : beta);
              tem += (C.at(mirrorIndex(i + 1, m), j) == k ? alpha : beta);
              tem += (C.at(i, mirrorIndex(j + 1, n)) == k ? alpha : beta);
              tem += (C.at(i, mirrorIndex(j - 1, n)) == k ? alpha : beta);
              
              Pot_f(k - 1) = 2 * tem;
            }
            Pro = Pot_f + logN.col(m * j + i);
            Pro = Pro - Pro.max();
            //cout <<Pro<<endl;
            Pro = exp(Pro);
            Pro = Pro / sum(Pro);
            distSampling = trng::discrete_dist(Pro.begin(), Pro.end());
            postC(i, j) = distSampling(rx);
          }
        }
      }
}

for (int i = 1; i <= K; i++)
{
  uvec cnt = find(postC == i);
  nk(i - 1) = cnt.n_elem;
}
        }
        C = postC;
        return postC;
      }