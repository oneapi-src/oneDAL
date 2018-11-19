/* file: zscore_dense_default_impl.i */
/*******************************************************************************
* Copyright 2014-2018 Intel Corporation.
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

template<typename algorithmFPType, CpuType cpu>
Status ZScoreKernel<algorithmFPType, defaultDense, cpu>::computeMeanVariance_thr
        (NumericTable& inputTable,
         algorithmFPType* resultMean,
         algorithmFPType* resultVariance,
         const daal::algorithms::Parameter& par)
{
    auto* parameter = static_cast<Parameter<algorithmFPType, defaultDense> *>(const_cast<daal::algorithms::Parameter *>(&par));
    return computeMeansAndVariances(parameter->moments.get(), inputTable, resultMean, resultVariance);
}

} // namespace internal
} // namespace zscore
} // namespace normalization
} // namespace algorithms
} // namespace daal

#endif
