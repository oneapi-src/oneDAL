/* file: zscore_dense_default_impl.i */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation
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
//  Implementation of defaultDense method for zscore normalization algorithm
//--
*/

#ifndef __ZSCORE_DENSE_DEFAULT_IMPL_I__
#define __ZSCORE_DENSE_DEFAULT_IMPL_I__

#include "zscore_moments.h"

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
template <typename algorithmFPType, CpuType cpu>
Status ZScoreKernel<algorithmFPType, defaultDense, cpu>::computeMeanVariance_thr(NumericTable & inputTable, algorithmFPType * resultMean,
                                                                                 algorithmFPType * resultVariance,
                                                                                 const daal::algorithms::Parameter & par)
{
    auto * parameter = static_cast<Parameter<algorithmFPType, defaultDense> *>(const_cast<daal::algorithms::Parameter *>(&par));
    return computeMeansAndVariances(parameter->moments.get(), inputTable, resultMean, resultVariance);
}

} // namespace internal
} // namespace zscore
} // namespace normalization
} // namespace algorithms
} // namespace daal

#endif
