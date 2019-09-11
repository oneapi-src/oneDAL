/* file: abs_base.h */
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
//  Declaration of template function that calculate abss.
//--


#ifndef __ABS_BASE_H__
#define __ABS_BASE_H__

#include "math/abs.h"
#include "kernel.h"
#include "numeric_table.h"
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
namespace math
{
namespace abs
{
namespace internal
{
/**
 *  \brief Kernel for abs calculation
 *  in case floating point type of intermediate calculations
 *  and method of calculations are different
 */
template<typename algorithmFPType, Method method, CpuType cpu>
class AbsKernelBase : public Kernel
{
public:
    Status compute(const NumericTable *inputTable, NumericTable *resultTable);

protected:
    static const size_t _nRowsInBlock = 5000;

    virtual Status processBlock(const NumericTable &inputTable, size_t nInputColumns, size_t nProcessedRows, size_t nRowsInCurrentBlock, NumericTable &resultTable) = 0;
};

template<typename algorithmFPType, Method method, CpuType cpu>
class AbsKernel {};

} // namespace daal::internal
} // namespace abs
} // namespace math
} // namespace algorithms
} // namespace daal

#endif
