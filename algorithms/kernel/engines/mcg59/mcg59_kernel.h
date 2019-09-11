/* file: mcg59_kernel.h */
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
//  Declaration of template function that calculate mcg59s.
//--

#ifndef __MCG59_KERNEL_H__
#define __MCG59_KERNEL_H__

#include "engines/mcg59/mcg59.h"
#include "kernel.h"
#include "numeric_table.h"

using namespace daal::services;
using namespace daal::data_management;

namespace daal
{
namespace algorithms
{
namespace engines
{
namespace mcg59
{
namespace internal
{
/**
 *  \brief Kernel for mcg59 calculation
 */
template<typename algorithmFPType, Method method, CpuType cpu>
class Mcg59Kernel : public Kernel
{
public:
    Status compute(NumericTable *resultTable);
};

} // namespace internal
} // namespace mcg59
} // namespace engines
} // namespace algorithms
} // namespace daal

#endif
