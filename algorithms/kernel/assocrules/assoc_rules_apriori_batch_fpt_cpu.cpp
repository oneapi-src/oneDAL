/* file: assoc_rules_apriori_batch_fpt_cpu.cpp */
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
//  Implementation of association rules mining algorithm.
//--
*/

#include "assoc_rules_batch_container.h"
#include "assoc_rules_apriori_kernel.h"
#include "assoc_rules_apriori_impl.i"

namespace daal
{
namespace algorithms
{
namespace association_rules
{

namespace interface1
{
template class BatchContainer<DAAL_FPTYPE, apriori, DAAL_CPU>;
} // namespace interface1

namespace internal
{
template class AssociationRulesKernel<apriori, DAAL_FPTYPE, DAAL_CPU>;
} // namespace internal

} // namespace association_rules
} // namespace algorithms
} // namespace daal
