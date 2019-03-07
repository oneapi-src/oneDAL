/* file: covariance_kernel.h */
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
//  Declaration of template structs that calculate Covariance matrix.
//--
*/


#ifndef __COVARIANCE_KERNEL_H__
#define __COVARIANCE_KERNEL_H__

#include "numeric_table.h"
#include "algorithm_base_common.h"
#include "covariance_types.h"

using namespace daal::services;
using namespace daal::data_management;

namespace daal
{
namespace algorithms
{
namespace covariance
{
namespace internal
{

template<typename algorithmFPType, Method method, CpuType cpu>
class CovarianceDenseBatchKernel : public daal::algorithms::Kernel
{
public:
    services::Status compute(NumericTable *dataTable, NumericTable *covTable,
                             NumericTable *meanTable, const Parameter *parameter);
};

template<typename algorithmFPType, Method method, CpuType cpu>
class CovarianceCSRBatchKernel : public daal::algorithms::Kernel
{
public:
    services::Status compute(NumericTable *dataTable, NumericTable *covTable,
                             NumericTable *meanTable, const Parameter *parameter);
};

template<typename algorithmFPType, Method method, CpuType cpu>
class CovarianceDenseOnlineKernel : public daal::algorithms::Kernel
{
public:
    services::Status compute(NumericTable *dataTable, NumericTable *nObsTable,
                             NumericTable *crossProductTable, NumericTable *sumTable,
                             const Parameter *parameter);

    services::Status finalizeCompute(NumericTable *nObsTable, NumericTable *crossProductTable,
                                     NumericTable *sumTable, NumericTable *covTable,
                                     NumericTable *meanTable, const Parameter *parameter);
};

template<typename algorithmFPType, Method method, CpuType cpu>
class CovarianceCSROnlineKernel : public daal::algorithms::Kernel
{
public:
    services::Status compute(NumericTable *dataTable, NumericTable *nObsTable,
                             NumericTable *crossProductTable, NumericTable *sumTable,
                             const Parameter *parameter);

    services::Status finalizeCompute(NumericTable *nObsTable, NumericTable *crossProductTable,
                                     NumericTable *sumTable, NumericTable *covTable,
                                     NumericTable *meanTable, const Parameter *parameter);
};

template<typename algorithmFPType, Method method, CpuType cpu>
class CovarianceDistributedKernel : public daal::algorithms::Kernel
{
public:
    services::Status compute(DataCollection *partialResultsCollection,
                             NumericTable *nObsTable, NumericTable *crossProductTable,
                             NumericTable *sumTable, const Parameter *parameter);

    services::Status finalizeCompute(NumericTable *nObsTable, NumericTable *crossProductTable,
                                     NumericTable *sumTable, NumericTable *covTable,
                                     NumericTable *meanTable, const Parameter *parameter);
};

} // namespace internal
} // namespace covariance
} // namespace algorithms
} // namespace daal

#endif
