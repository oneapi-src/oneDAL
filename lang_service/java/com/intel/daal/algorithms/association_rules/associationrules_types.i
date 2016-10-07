/* file: associationrules_types.i */
/*******************************************************************************
* Copyright 2014-2016 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
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
