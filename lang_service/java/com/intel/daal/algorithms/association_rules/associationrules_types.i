/* file: associationrules_types.i */
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

#include "daal.h"

#include "JComputeMode.h"
#include "association_rules/JMethod.h"
#include "association_rules/JInputId.h"
#include "association_rules/JItemsetsOrderId.h"
#include "association_rules/JRulesOrderId.h"
#include "association_rules/JResultId.h"

using namespace daal;
using namespace daal::algorithms;

#define jBatch   com_intel_daal_algorithms_ComputeMode_batchValue

#define Apriori com_intel_daal_algorithms_association_rules_Method_Apriori

#define ItemsetsUnsorted        com_intel_daal_algorithms_association_rules_ItemsetsOrderId_ItemsetsUnsorted
#define ItemsetsSortedBySupport com_intel_daal_algorithms_association_rules_ItemsetsOrderId_ItemsetsSortedBySupport

#define RulesUnsorted           com_intel_daal_algorithms_association_rules_RulesOrderId_RulesUnsorted
#define RulesSortedByConfidence com_intel_daal_algorithms_association_rules_RulesOrderId_RulesSortedByConfidence

#define LargeItemsets        com_intel_daal_algorithms_association_rules_ResultId_LargeItemsets
#define LargeItemsetsSupport com_intel_daal_algorithms_association_rules_ResultId_LargeItemsetsSupport
#define AntecedentItemsets   com_intel_daal_algorithms_association_rules_ResultId_AntecedentItemsets
#define ConsequentItemsets   com_intel_daal_algorithms_association_rules_ResultId_ConsequentItemsets
#define Confidence           com_intel_daal_algorithms_association_rules_ResultId_Confidence

typedef association_rules::Batch<float, association_rules::apriori>     ar_of_s_ap;
typedef association_rules::Batch<double, association_rules::apriori>    ar_of_d_ap;
typedef services::SharedPtr<association_rules::Batch<float, association_rules::apriori> >    sp_ar_of_s_ap;
typedef services::SharedPtr<association_rules::Batch<double, association_rules::apriori> >   sp_ar_of_d_ap;
