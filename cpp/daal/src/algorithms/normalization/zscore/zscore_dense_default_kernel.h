/* file: zscore_dense_default_kernel.h */
/*******************************************************************************
* Copyright 2014-2022 Intel Corporation
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

#ifndef __ZSCORE_DENSE_DEFAULT_KERNEL_H__
#define __ZSCORE_DENSE_DEFAULT_KERNEL_H__

#include "src/externals/service_math.h"

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
 *  \brief Specialization of the structure that contains kernels for z-score normalization using defaultDense method
 */
template <typename algorithmFPType, CpuType cpu>
class ZScoreKernel<algorithmFPType, defaultDense, cpu> : public ZScoreKernelBase<algorithmFPType, cpu>
{
public:
    Status computeMeanVariance_thr(NumericTable & inputTable, algorithmFPType * resultMean, algorithmFPType * resultVariance,
                                   const daal::algorithms::Parameter & parameter) DAAL_C11_OVERRIDE;
};

} // namespace internal
} // namespace zscore
} // namespace normalization
} // namespace algorithms
} // namespace daal

#endif
