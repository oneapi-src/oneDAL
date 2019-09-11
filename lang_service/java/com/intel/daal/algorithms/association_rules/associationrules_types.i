/* file: associationrules_types.i */
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

#include "daal.h"

using namespace daal;
using namespace daal::algorithms;

#include "com_intel_daal_algorithms_association_rules_ItemsetsOrderId.h"
#define ItemsetsUnsorted        com_intel_daal_algorithms_association_rules_ItemsetsOrderId_ItemsetsUnsorted
#define ItemsetsSortedBySupport com_intel_daal_algorithms_association_rules_ItemsetsOrderId_ItemsetsSortedBySupport

#include "com_intel_daal_algorithms_association_rules_RulesOrderId.h"
#define RulesUnsorted           com_intel_daal_algorithms_association_rules_RulesOrderId_RulesUnsorted
#define RulesSortedByConfidence com_intel_daal_algorithms_association_rules_RulesOrderId_RulesSortedByConfidence

typedef association_rules::Batch<float, association_rules::apriori>     ar_of_s_ap;
typedef association_rules::Batch<double, association_rules::apriori>    ar_of_d_ap;
typedef services::SharedPtr<association_rules::Batch<float, association_rules::apriori> >    sp_ar_of_s_ap;
typedef services::SharedPtr<association_rules::Batch<double, association_rules::apriori> >   sp_ar_of_d_ap;
