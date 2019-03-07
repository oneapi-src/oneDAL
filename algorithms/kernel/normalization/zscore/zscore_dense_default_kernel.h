/* file: zscore_dense_default_kernel.h */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation.
*
* This software and the related documents are Intel copyrighted  materials,  and
* your use of  them is  governed by the  express license  under which  they were
* provided to you (License).  Unless the License provides otherwise, you may not
* use, modify, copy, publish, distribute,  disclose or transmit this software or
* the related documents without Intel's prior written permission.
*
* This software and the related documents  are provided as  is,  with no express
* or implied  warranties,  other  than those  that are  expressly stated  in the
* License.
*******************************************************************************/

/*
//++
//  Implementation of defaultDense method for zscore normalization algorithm
//--
*/

#ifndef __ZSCORE_DENSE_DEFAULT_KERNEL_H__
#define __ZSCORE_DENSE_DEFAULT_KERNEL_H__

#include "service_math.h"

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
template<typename algorithmFPType, CpuType cpu>
class ZScoreKernel<algorithmFPType, defaultDense, cpu> : public ZScoreKernelBase<algorithmFPType, cpu>
{
public:
    Status computeMeanVariance_thr(NumericTable& inputTable,
                                   algorithmFPType* resultMean,
                                   algorithmFPType* resultVariance,
                                   const daal::algorithms::Parameter& parameter)  DAAL_C11_OVERRIDE;
};

} // namespace daal::internal
} // namespace zscore
} // namespace normalization
} // namespace algorithms
} // namespace daal

#endif
