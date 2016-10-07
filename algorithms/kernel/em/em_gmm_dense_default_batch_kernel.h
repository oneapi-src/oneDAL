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
#include "em_gmm_dense_default_batch_task.h"

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
    void setVariables(const NumericTable *a, const Parameter *par);
    void getInitValues(NumericTable *initialWeights, NumericTable *initialMeans, NumericTable **initialCovariances, GmmModel<algorithmFPType, cpu> *covs);

    void stepM_merge(size_t iteration, GmmModel<algorithmFPType, cpu> *covs);

    algorithmFPType *alpha;
    algorithmFPType *means;

    algorithmFPType *logAlpha;

    size_t blockSizeDefault;
    size_t nBlocks;

    double covRegularizer;

    DAAL_INT nFeatures;
    DAAL_INT nVectors;
    DAAL_INT nComponents;
    algorithmFPType logLikelyhoodCorrection;
    DAAL_INT maxIterations;
    algorithmFPType threshold;

public:
    void compute(
        NumericTable *dataTable,
        NumericTable *initialWeights,
        NumericTable *initialMeans,
        NumericTable **initialCovariances,
        NumericTable *resultWeights,
        NumericTable *resultMeans,
        NumericTable **resultCovariances,
        NumericTable *resultNIterations,
        NumericTable *resultGoalFunction,
        const Parameter *par);
};


template<typename algorithmFPType, CpuType cpu>
void stepM_mergePartialSums(
    algorithmFPType *cp_n, algorithmFPType *cp_m,
    algorithmFPType *mean_n, algorithmFPType *mean_m,
    algorithmFPType &w_n, algorithmFPType &w_m,
    size_t nFeatures, GmmModel<algorithmFPType, cpu> *covs);

} // namespace internal

} // namespace em_gmm

} // namespace algorithms

} // namespace daal

#endif
