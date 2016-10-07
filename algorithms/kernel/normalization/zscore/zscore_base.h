/* file: zscore_base.h */
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

//++
//  Declaration of template function that calculates zscore normalization.
//--

#ifndef __ZSCORE_BASE_H__
#define __ZSCORE_BASE_H__

#include "normalization/zscore.h"
#include "kernel.h"
#include "numeric_table.h"

using namespace daal::data_management;
using namespace daal::services;

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
template<typename algorithmFPType, CpuType cpu>
class ZScoreKernelBase : public Kernel
{
public:

    /**
     *  \brief Function that computes z-score normalization
     *
     *  \param input[in]        Input of the algorithm
     *  \param result[out]      Result of the algorithm
     *  \param parameter[in]    Parameters of the algorithm
     */
    void compute(SharedPtr<NumericTable> inputTable,  NumericTable *sumTable, NumericTable *resultTable, daal::algorithms::Parameter *parameter);

    virtual int computeMeanVariance_thr( SharedPtr<NumericTable> inputTable,
                                        algorithmFPType* resultMean,
                                        algorithmFPType* resultVariance,
                                        daal::algorithms::Parameter *parameter
                                         ) = 0;

};

template <typename algorithmFPType, Method method, CpuType cpu>
class ZScoreKernel : public ZScoreKernelBase<algorithmFPType, cpu>
{};

} // namespace daal::internal
} // namespace zscore
} // namespace normalization
} // namespace algorithms
} // namespace daal

#include "zscore_impl.i"

#endif
