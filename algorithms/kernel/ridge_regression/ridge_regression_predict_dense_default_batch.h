/* file: ridge_regression_predict_dense_default_batch.h */
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
//  Declaration of template function that computes ridge regression prediction results.
//--
*/

#ifndef __RIDGE_REGRESSION_PREDICT_DENSE_DEFAULT_BATCH_H__
#define __RIDGE_REGRESSION_PREDICT_DENSE_DEFAULT_BATCH_H__

#include "ridge_regression_predict.h"
#include "service_memory.h"
#include "kernel.h"
#include "numeric_table.h"
#include "service_blas.h"

using namespace daal::data_management;

namespace daal
{
namespace algorithms
{
namespace ridge_regression
{
namespace prediction
{
namespace internal
{

template <typename algorithmFpType, prediction::Method method, CpuType cpu>
class RidgeRegressionPredictKernel : public daal::algorithms::Kernel
{
public:
    /**
     *  \brief Compute ridge regression prediction results.
     *
     *  \param a[in]    Matrix of input variables X
     *  \param m[in]    Ridge regression model obtained on training stage
     *  \param r[out]   Prediction results
     *  \param par[in]  Ridge regression algorithm parameters
     */
    void compute(const NumericTable *a, const daal::algorithms::Model *m, NumericTable *r, const daal::algorithms::Parameter *par);
};

template <typename algorithmFpType, CpuType cpu>
class RidgeRegressionPredictKernel<algorithmFpType, defaultDense, cpu> : public daal::algorithms::Kernel
{
public:
    void compute(const NumericTable *a, const daal::algorithms::Model *m, NumericTable *r, const daal::algorithms::Parameter *par);

protected:
    void computeBlockOfResponses(MKL_INT *numFeatures, MKL_INT *numRows, algorithmFpType *dataBlock, MKL_INT *numBetas, algorithmFpType *beta,
                                 MKL_INT *numResponses, algorithmFpType *responseBlock, bool findBeta0);
};

} // namespace internal
} // namespace prediction
} // namespace ridge_regression
} // namespace algorithms
} // namespace daal

#endif // __RIDGE_REGRESSION_PREDICT_DENSE_DEFAULT_BATCH_H__
