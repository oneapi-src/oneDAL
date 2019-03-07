@echo off
rem ============================================================================
rem Copyright 2017-2019 Intel Corporation.
rem
rem This software  and the related  documents  are Intel  copyrighted materials,
rem and your use  of them is  governed by the express  license under  which they
rem were provided to you (License).  Unless the License provides otherwise,  you
rem may not use,  modify,  copy, publish,  distribute, disclose or transmit this
rem software or the related documents without Intel's prior written permission.
rem
rem This software and the related documents are provided as is,  with no express
rem or implied warranties,  other  than those that  are expressly  stated in the
rem License.
rem
rem License:
rem http://software.intel.com/en-us/articles/intel-sample-source-code-license-a
rem greement/
rem ============================================================================

::  Content:
::     Intel(R) Data Analytics Acceleration Library samples list
::******************************************************************************

set MPI_SAMPLE_LIST=svd_fast_distributed_mpi                      ^
                    qr_fast_distributed_mpi                       ^
                    linear_regression_norm_eq_distributed_mpi     ^
                    linear_regression_qr_distributed_mpi          ^
                    pca_correlation_dense_distributed_mpi         ^
                    pca_correlation_csr_distributed_mpi           ^
                    pca_svd_distributed_mpi                       ^
                    covariance_dense_distributed_mpi              ^
                    covariance_csr_distributed_mpi                ^
                    multinomial_naive_bayes_dense_distributed_mpi ^
                    multinomial_naive_bayes_csr_distributed_mpi   ^
                    kmeans_dense_distributed_mpi                  ^
                    kmeans_csr_distributed_mpi                    ^
                    kmeans_init_dense_distributed_mpi             ^
                    kmeans_init_csr_distributed_mpi               ^
                    low_order_moments_csr_distributed_mpi         ^
                    low_order_moments_dense_distributed_mpi       ^
                    implicit_als_csr_distributed_mpi              ^
                    ridge_regression_norm_eq_distributed_mpi      ^
                    neural_net_dense_distributed_mpi              ^
                    neural_net_dense_allgather_distributed_mpi    ^
                    neural_net_dense_asynch_distributed_mpi
