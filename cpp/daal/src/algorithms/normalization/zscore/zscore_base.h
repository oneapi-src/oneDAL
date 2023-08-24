/* file: zscore_base.h */
/*******************************************************************************
* Copyright 2014 Intel Corporation
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

//++
//  Declaration of template function that calculates zscore normalization.
//--

#ifndef __ZSCORE_BASE_H__
#define __ZSCORE_BASE_H__

#include "algorithms/normalization/zscore_types.h"
#include "src/algorithms/kernel.h"
#include "data_management/data/numeric_table.h"
#include "src/externals/service_math.h"
#include "src/data_management/service_numeric_table.h"
#include "src/algorithms/service_error_handling.h"
#include "src/threading/threading.h"

using namespace daal::data_management;
using namespace daal::services;
using namespace daal::internal;

namespace daal
{
namespace algorithms
{
namespace normalization
{
namespace zscore
{
namespace internal
{
/**
 *  \brief Kernel for zscore normalization calculation
 *  in case floating point type of intermediate calculations
 *  and method of calculations are different
 */
template <typename algorithmFPType, CpuType cpu>
class ZScoreKernelBase : public Kernel
{
public:
    /**
    *  \brief Function that computes z-score normalization
    *
    *  \param inputTable[in]    Input data of the algorithm
    *  \param resultTable[out]  Table that stores algotithm's results
    *  \param parameter[in]     Parameters of the algorithm
    */
    Status compute(NumericTable & inputTable, NumericTable & resultTable, const daal::algorithms::Parameter & parameter);

    /**
     *  \brief Function that computes z-score normalization
     *
     *  \param inputTable[in]    Input data of the algorithm
     *  \param resultTable[out]  Table that stores normalized data results
     *  \param resultMeans[out]  Table that stores means results
     *  \param resultVariances[out]  Table that stores variances results
     *  \param parameter[in]     Parameters of the algorithm
     */
    Status compute(NumericTable & inputTable, NumericTable & resultTable, NumericTable & resultMeans, NumericTable & resultVariances,
                   const daal::algorithms::Parameter & parameter);

protected:
    Status common_compute(NumericTable & inputTable, NumericTable & resultTable, algorithmFPType * means_total, algorithmFPType * variances_total,
                          const daal::algorithms::Parameter & parameter);

    virtual Status computeMeanVariance_thr(NumericTable & inputTable, algorithmFPType * resultMean, algorithmFPType * resultVariance,
                                           const daal::algorithms::Parameter & parameter) = 0;
};

template <typename algorithmFPType, Method method, CpuType cpu>
class ZScoreKernel : public ZScoreKernelBase<algorithmFPType, cpu>
{};

} // namespace internal
} // namespace zscore
} // namespace normalization
} // namespace algorithms
} // namespace daal

#include "src/algorithms/normalization/zscore/zscore_impl.i"

#endif
