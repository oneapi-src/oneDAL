/* file: ridge_regression_train_kernel.h */
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
//  Declaration of structure containing kernels for ridge regression training.
//--
*/

#ifndef __RIDGE_REGRESSION_TRAIN_KERNEL_H__
#define __RIDGE_REGRESSION_TRAIN_KERNEL_H__

#include "numeric_table.h"
#include "algorithm_base_common.h"
#include "ridge_regression_training_types.h"
#include "service_lapack.h"

namespace daal
{
namespace algorithms
{
namespace ridge_regression
{

using namespace daal::data_management;
using namespace daal::services;

namespace training
{
namespace internal
{

template <typename algorithmFpType, training::Method method, CpuType cpu>
class RidgeRegressionTrainBatchKernel
{};

template <typename algorithmFpType, CpuType cpu>
class RidgeRegressionTrainBatchKernel<algorithmFpType, training::normEqDense, cpu> : public daal::algorithms::Kernel
{
public:
    void compute(NumericTable *x, NumericTable *y, ridge_regression::Model *r,
                 const daal::algorithms::Parameter * par);
};

template <typename algorithmfptype, training::Method method, CpuType cpu>
class RidgeRegressionTrainOnlineKernel
{};

template <typename algorithmFpType, CpuType cpu>
class RidgeRegressionTrainOnlineKernel<algorithmFpType, training::normEqDense, cpu> : public daal::algorithms::Kernel
{
public:
    void compute(NumericTable *x, NumericTable *y, ridge_regression::Model *r,
                 const daal::algorithms::Parameter * par);

    void finalizeCompute(ridge_regression::Model *a, ridge_regression::Model *r,
                         const daal::algorithms::Parameter * par);
};

template <typename algorithmFpType, training::Method method, CpuType cpu>
class RidgeRegressionTrainDistributedKernel
{};

template <typename algorithmFpType, CpuType cpu>
class RidgeRegressionTrainDistributedKernel<algorithmFpType, training::normEqDense, cpu> : public daal::algorithms::Kernel
{
public:
    void compute(size_t n, daal::algorithms::Model ** partialModels, daal::algorithms::Model * r, const daal::algorithms::Parameter * par);

    void finalizeCompute(ridge_regression::Model *a, ridge_regression::Model *r, const daal::algorithms::Parameter * par);

protected:
    void merge(daal::algorithms::Model * a, daal::algorithms::Model * r, const daal::algorithms::Parameter * par);

    void mergePartialSums(MKL_INT dim, MKL_INT ny, algorithmFpType * axtx, algorithmFpType * axty, algorithmFpType * rxtx, algorithmFpType * rxty);
};

} // namespace internal
} // namespace training
} // namespace ridge_regression
} // namespace algorithms
} // namespace daal

#endif
