/* file: zscore_base.h */
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

//++
//  Declaration of template function that calculates zscore normalization.
//--

#ifndef __ZSCORE_BASE_H__
#define __ZSCORE_BASE_H__

#include "inner/zscore_v1.h"
#include "kernel.h"
#include "numeric_table.h"
#include "service_math.h"
#include "service_numeric_table.h"
#include "service_error_handling.h"
#include "threading.h"

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
template<typename algorithmFPType, CpuType cpu>
class ZScoreKernelBase : public Kernel
{
public:

    /**
    *  \brief Function that computes z-score normalization for interface1
    *
    *  \param inputTable[in]    Input data of the algorithm
    *  \param resultTable[out]  Table that stores algotithm's results
    *  \param parameter[in]     Parameters of the algorithm
    */
    Status compute(NumericTable &inputTable, NumericTable &resultTable, const daal::algorithms::Parameter &parameter);

    /**
     *  \brief Function that computes z-score normalization
     *
     *  \param inputTable[in]    Input data of the algorithm
     *  \param resultTable[out]  Table that stores normalized data results
     *  \param resultMeans[out]  Table that stores means results
     *  \param resultVariances[out]  Table that stores variances results
     *  \param parameter[in]     Parameters of the algorithm
     */
    Status compute(NumericTable &inputTable, NumericTable &resultTable,
                   NumericTable &resultMeans,
                   NumericTable &resultVariances,
                   const daal::algorithms::Parameter &parameter);

protected:
    Status common_compute(NumericTable &inputTable,
                          NumericTable &resultTable,
                          algorithmFPType* means_total,
                          algorithmFPType* variances_total,
                          const daal::algorithms::Parameter &parameter);

    virtual Status computeMeanVariance_thr(NumericTable &inputTable,
                                           algorithmFPType* resultMean,
                                           algorithmFPType* resultVariance,
                                           const daal::algorithms::Parameter &parameter) = 0;
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
