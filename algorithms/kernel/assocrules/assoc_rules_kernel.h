/* file: assoc_rules_kernel.h */
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
//  Declaration of template function that computes association rules results.
//--
*/

#ifndef __ASSOC_RULES_KERNEL_H__
#define __ASSOC_RULES_KERNEL_H__

#include "apriori_types.h"
#include "kernel.h"

#include "numeric_table.h"

using namespace daal::data_management;

namespace daal
{
namespace algorithms
{
namespace association_rules
{
namespace internal
{

/**
 *  Structure that contains kernels for association rules mining
 */
template <Method method, typename algorithmFPType, CpuType cpu>
class AssociationRulesKernel : public Kernel
{
public:
    /** Find "large" item sets and build association rules */
    services::Status compute(const NumericTable *a, NumericTable *r[], const daal::algorithms::Parameter *parameter);
};

} // namespace internal

} // namespace association_rules

} // namespace algorithms

} // namespace daal

#endif
