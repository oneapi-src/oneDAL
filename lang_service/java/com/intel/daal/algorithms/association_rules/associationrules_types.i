/* file: associationrules_types.i */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation
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

using namespace daal;
using namespace daal::algorithms;

#include "com_intel_daal_algorithms_association_rules_ItemsetsOrderId.h"
#define ItemsetsUnsorted        com_intel_daal_algorithms_association_rules_ItemsetsOrderId_ItemsetsUnsorted
#define ItemsetsSortedBySupport com_intel_daal_algorithms_association_rules_ItemsetsOrderId_ItemsetsSortedBySupport

#include "com_intel_daal_algorithms_association_rules_RulesOrderId.h"
#define RulesUnsorted           com_intel_daal_algorithms_association_rules_RulesOrderId_RulesUnsorted
#define RulesSortedByConfidence com_intel_daal_algorithms_association_rules_RulesOrderId_RulesSortedByConfidence

typedef association_rules::Batch<float, association_rules::apriori> ar_of_s_ap;
typedef association_rules::Batch<double, association_rules::apriori> ar_of_d_ap;
typedef services::SharedPtr<association_rules::Batch<float, association_rules::apriori> > sp_ar_of_s_ap;
typedef services::SharedPtr<association_rules::Batch<double, association_rules::apriori> > sp_ar_of_d_ap;
