/* file: linear_regression_train_kernel.h */
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
//  Declaration of structure containing kernels for linear regression
//  training.
//--
*/

#ifndef __LINEAR_REGRESSION_TRAIN_KERNEL_H__
#define __LINEAR_REGRESSION_TRAIN_KERNEL_H__

#include "numeric_table.h"
#include "algorithm_base_common.h"
#include "linear_regression_training_types.h"
#include "service_lapack.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace linear_regression
{
namespace training
{
namespace internal
{

template <typename interm, training::Method method, CpuType cpu>
class LinearRegressionTrainBatchKernel
{};

template <typename interm, CpuType cpu>
class LinearRegressionTrainBatchKernel<interm, training::normEqDense, cpu> : public daal::algorithms::Kernel
{
public:
    void compute(NumericTable *x, NumericTable *y, linear_regression::Model *r,
                 const daal::algorithms::Parameter *par);
};

template <typename interm, CpuType cpu>
class LinearRegressionTrainBatchKernel<interm, training::qrDense, cpu> : public daal::algorithms::Kernel
{
public:
    void compute(NumericTable *x, NumericTable *y, linear_regression::Model *r,
                 const daal::algorithms::Parameter *par);
};


template <typename interm, training::Method method, CpuType cpu>
class LinearRegressionTrainOnlineKernel
{};

template <typename interm, CpuType cpu>
class LinearRegressionTrainOnlineKernel<interm, training::normEqDense, cpu> : public daal::algorithms::Kernel
{
public:
    void compute(NumericTable *x, NumericTable *y, linear_regression::Model *r,
                 const daal::algorithms::Parameter *par);
    void finalizeCompute(linear_regression::Model *a, linear_regression::Model *r,
                         const daal::algorithms::Parameter *par);
};

template <typename interm, CpuType cpu>
class LinearRegressionTrainOnlineKernel<interm, training::qrDense, cpu> : public daal::algorithms::Kernel
{
public:
    void compute(NumericTable *x, NumericTable *y, linear_regression::Model *r,
                 const daal::algorithms::Parameter *par);
    void finalizeCompute(linear_regression::Model *a, linear_regression::Model *r,
                         const daal::algorithms::Parameter *par);
};


template <typename interm, training::Method method, CpuType cpu>
class LinearRegressionTrainDistributedKernel
{};

template <typename interm, CpuType cpu>
class LinearRegressionTrainDistributedKernel<interm, training::normEqDense, cpu> : public daal::algorithms::Kernel
{
public:
    void compute(size_t n, daal::algorithms::Model **partialModels, daal::algorithms::Model *r,
                 const daal::algorithms::Parameter *par);
    void finalizeCompute(linear_regression::Model *a, linear_regression::Model *r,
                         const daal::algorithms::Parameter *par);
protected:
    void merge(daal::algorithms::Model *a, daal::algorithms::Model *r, const daal::algorithms::Parameter *par);
    void mergePartialSums(MKL_INT dim, MKL_INT ny, interm *axtx, interm *axty, interm *rxtx, interm *rxty);
};

template <typename interm, CpuType cpu>
class LinearRegressionTrainDistributedKernel<interm, training::qrDense, cpu> : public daal::algorithms::Kernel
{
public:
    void compute(size_t n, daal::algorithms::Model **partialModels, daal::algorithms::Model *r,
                 const daal::algorithms::Parameter *par);
    void finalizeCompute(linear_regression::Model *a, linear_regression::Model *r,
                         const daal::algorithms::Parameter *par);
protected:
    void merge(daal::algorithms::Model *a, daal::algorithms::Model *r, const daal::algorithms::Parameter *par);
};
} // namespace internal
}
}
}
} // namespace daal


#endif
