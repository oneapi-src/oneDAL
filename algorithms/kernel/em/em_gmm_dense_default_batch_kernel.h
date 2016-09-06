/* file: em_gmm_dense_default_batch_kernel.h */
/*******************************************************************************
* Copyright 2014-2016 Intel Corporation
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
//  Declaration of template function that calculate ems.
//--
*/

#ifndef __EM_GMM_DENSE_DEFAULT_BATCH_KERNEL_H__
#define __EM_GMM_DENSE_DEFAULT_BATCH_KERNEL_H__

#include "em_gmm.h"
#include "kernel.h"
#include "numeric_table.h"
#include "service_blas.h"

using namespace daal::data_management;

namespace daal
{
namespace algorithms
{
namespace em_gmm
{
namespace internal
{

template<typename algorithmFPType, Method method, CpuType cpu>
class EMKernel : public Kernel
{
public:
    EMKernel() {};

private:
    void allocateMemory();
    void deallocate();

    void setVariables(const NumericTable *a, const size_t nr, const Parameter *par);
    void stepM_merge(algorithmFPType *v);
    void stepM_merge_inner(algorithmFPType *cp_n, algorithmFPType *cp_m, algorithmFPType *mean_n, algorithmFPType *mean_m, algorithmFPType &W_n,
                           algorithmFPType &W_m);
    void computeSigmaValues(size_t iteration);

    void writeResult(const size_t nr, NumericTable *r[]);
    void writeArrayToNumericTable(NumericTable *nt, algorithmFPType *array, size_t nColsArr, size_t nRowsArr);
    void getArrayFromNumericTable(const NumericTable *ntConst, algorithmFPType *array, size_t nColsArr, size_t nRowsArr);
    void getInitValues(NumericTable *r[]);
    void regularizeCovarianceMatrix(algorithmFPType *cov, services::Error * error);


    algorithmFPType *alpha;
    algorithmFPType *means;
    algorithmFPType *sigma;

    algorithmFPType *logSqrtInvDetSigma;
    algorithmFPType *logAlpha;

    algorithmFPType *buffer;

    size_t blockSizeDeafult;
    size_t memorySizeForOneThread;
    size_t memorySizeForOneBlockResult;
    size_t nBlocks;

    MKL_INT nFeatures;
    MKL_INT nVectors;
    MKL_INT nComponents;
    algorithmFPType logLikelyhoodCorrection;
    size_t iterCounter;
    MKL_INT maxIterations;
    algorithmFPType threshold;
    algorithmFPType logLikelyhood;

public:
    void compute(const size_t na, const NumericTable *const *a, const size_t nr, NumericTable *r[], const Parameter *par);
};

} // namespace internal

} // namespace em_gmm

} // namespace algorithms

} // namespace daal

#endif
