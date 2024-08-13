/* file: spectral_embedding_default_dense_impl.i */
/*******************************************************************************
* Copyright 2024 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

/*
//++
//  Implementation of cosine distance.
//--
*/

#include "services/daal_defines.h"
#include "src/externals/service_math.h"
#include "src/externals/service_blas.h"
#include "src/threading/threading.h"
#include "src/algorithms/service_error_handling.h"
#include "src/data_management/service_numeric_table.h"
#include "src/algorithms/cosdistance/cosdistance_kernel.h"
#include "src/externals/service_lapack.h"
#include <iostream>


using namespace daal::internal;

namespace daal
{
namespace algorithms
{
namespace spectral_embedding
{
namespace internal
{



template <typename algorithmFPType, CpuType cpu>
services::Status computeEigenvectorsInplace(size_t nFeatures, algorithmFPType * eigenvectors, algorithmFPType * eigenvalues)
{
    char jobz = 'V';
    char uplo = 'U';

    DAAL_INT lwork  = 2 * nFeatures * nFeatures + 6 * nFeatures + 1;
    DAAL_INT liwork = 5 * nFeatures + 3;
    DAAL_INT info;

    TArray<algorithmFPType, cpu> work(lwork);
    TArray<DAAL_INT, cpu> iwork(liwork);
    DAAL_CHECK_MALLOC(work.get() && iwork.get());

    LapackInst<algorithmFPType, cpu>::xsyevd(&jobz, &uplo, (DAAL_INT *)(&nFeatures), eigenvectors, (DAAL_INT *)(&nFeatures), eigenvalues, work.get(),
                                             &lwork, iwork.get(), &liwork, &info);
    if (info != 0) return services::Status(services::ErrorPCAFailedToComputeCorrelationEigenvalues); // CHANGE ERROR STATUS
    return services::Status();
}


/**
 *  \brief Kernel for Spectral Embedding calculation
 */
template <typename algorithmFPType, Method method, CpuType cpu>
services::Status SpectralEmbeddingKernel<algorithmFPType, method, cpu>::compute(const NumericTable* xTable, NumericTable* eTable, const KernelParameter & par)
{
    
    services::Status safeStat{};
    std::cout << "inside DAAL kernel" << std::endl;


    std::cout << "Params: " << par.num_emb << " " << par.p << std::endl;
    int filt_num = par.p;

    //size_t na = input->size();
    //size_t nr = result->size();

    NumericTable * a0                      = const_cast<NumericTable *>(xTable);
    NumericTable ** a                      = &a0;
    // NumericTable * r0                      = static_cast<NumericTable *>(result->get(cosineDistance).get());
    NumericTable ** r                      = &eTable;
    // _env = daal::services::
    // daal::services::Environment::env & env = *_env;

    // services::Status cos_dist_status = cosine_distance::internal::DistanceKernel<algorithmFPType, cosine_distance::Method::defaultDense, cpu>::compute(0, a, 0, r, nullptr);

    //__DAAL_CALL_KERNEL(env, cosine_distance::internal::DistanceKernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, cosine_distance::Method::defaultDense), compute, 0, a, 0, r, nullptr);

    cosine_distance::internal::DistanceKernel<algorithmFPType, cosine_distance::Method::defaultDense, cpu>* cos_dist_kernel_ptr = new cosine_distance::internal::DistanceKernel<algorithmFPType, cosine_distance::Method::defaultDense, cpu>();
    
    services::Status cos_dist_status = cos_dist_kernel_ptr->compute(0, a, 0, r, nullptr);

    delete cos_dist_kernel_ptr;

    size_t n = xTable->getNumberOfRows();    /* Number of input feature vectors   */

    for (int i = 0; i < n; ++i) {
        algorithmFPType L = 0;
        int lcnt = 0;
        algorithmFPType R = 2;
        int rcnt = n;
        int cnt;
        std::cout << "Row " << i << std::endl;
        WriteRows<algorithmFPType, cpu> xBlock(*r, i, 1);
        DAAL_CHECK_BLOCK_STATUS(xBlock);
        algorithmFPType * x = xBlock.get();
        for (int j = 0; j < n; ++j) {
            std::cout << x[j] << " ";
        }
        std::cout << std::endl;
        for (int ij = 0; ij < 20; ++ij) {
            algorithmFPType M = (L + R) / 2;
            cnt = 0;
            for (int j = 0; j < n; ++j) {
                if (x[j] <= M) {
                    cnt++;
                }
            }
            if (cnt < filt_num) {
                L = M;
                lcnt = cnt;
            } else {
                R = M;
                rcnt = cnt;
            }
            if (lcnt + 1 == rcnt) {
                break;
            }
        }
        for (int j = 0; j < n; ++j) {
            if (x[j] <= R) {
                x[j] = 1.0;
            } else {
                x[j] = 0.0;
            }
        }
        x[i] = 0;
    }
    WriteRows<algorithmFPType, cpu> xMatrix(*r, 0, n);
    DAAL_CHECK_BLOCK_STATUS(xMatrix);
    algorithmFPType * x = xMatrix.get();

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < i; ++j) {
            algorithmFPType val = (x[i * n +j] + x[j * n + i]) / 2;
            x[i * n + j] = -val;
            x[j * n + i] = -val; 
            x[i * n + i] += val;
            x[j * n + j] += val; 
        }
    }

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            std::cout << x[i * n + j] << " ";
        }
        std::cout << std::endl;
    }


    TArray<algorithmFPType, cpu> eigenvalues(n);
    DAAL_CHECK_MALLOC(eigenvalues.get());

    services::Status eigen_vectors_st = computeEigenvectorsInplace<algorithmFPType, cpu>(n, x, eigenvalues.get());
    if (!eigen_vectors_st) {
        return eigen_vectors_st;
    }

    return services::Status{};
    
    // NumericTable * xTable                          = const_cast<NumericTable *>(a[0]); /* Input data */
    // NumericTable * rTable                          = const_cast<NumericTable *>(r[0]); /* Output data */
    // const NumericTableIface::StorageLayout rLayout = r[0]->getDataLayout();

    // if (isFull<algorithmFPType, cpu>(rLayout))
    // {
    //     return cosDistanceFull<algorithmFPType, cpu>(xTable, rTable);
    // }
    // else
    // {
    //     if (isLower<algorithmFPType, cpu>(rLayout))
    //     {
    //         return cosDistanceLowerPacked<algorithmFPType, cpu>(xTable, rTable);
    //     }
    //     else if (isUpper<algorithmFPType, cpu>(rLayout))
    //     {
    //         return cosDistanceUpperPacked<algorithmFPType, cpu>(xTable, rTable);
    //     }
    //     else
    //     {
    //         return services::Status(services::ErrorIncorrectTypeOfOutputNumericTable);
    //     }
    // }
    
}

} // namespace internal

} // namespace cosine_distance

} // namespace algorithms

} // namespace daal
