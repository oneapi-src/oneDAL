/* file: df_classification_training_types_v1.cpp */
/*******************************************************************************
* Copyright 2014-2021 Intel Corporation
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

/*
//++
//  Implementation of decision forest algorithm classes.
//--
*/

#include "src/algorithms/dtrees/forest/classification/df_classification_training_types_result.h"
#include "src/services/serialization_utils.h"
#include "src/services/daal_strings.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace decision_forest
{
namespace classification
{
namespace training
{
namespace interface1
{
DAAL_FORCEINLINE void convertParameter(const interface1::Parameter & par1, interface3::Parameter & par3)
{
    par3.nTrees                      = par1.nTrees;
    par3.observationsPerTreeFraction = par1.observationsPerTreeFraction;
    par3.featuresPerNode             = par1.featuresPerNode;
    par3.maxTreeDepth                = par1.maxTreeDepth;
    par3.minObservationsInLeafNode   = par1.minObservationsInLeafNode;
    par3.seed                        = par1.seed;
    par3.engine                      = par1.engine;
    par3.impurityThreshold           = par1.impurityThreshold;
    par3.varImportance               = par1.varImportance;
    par3.resultsToCompute            = par1.resultsToCompute;
    par3.memorySavingMode            = par1.memorySavingMode;
    par3.bootstrap                   = par1.bootstrap;
}

services::Status Parameter::check() const
{
    services::Status s;
    DAAL_CHECK_STATUS(s, classifier::interface1::Parameter::check());

    decision_forest::classification::training::interface3::Parameter par3(this->nClasses);
    convertParameter(*this, par3);

    DAAL_CHECK_STATUS(s, decision_forest::training::checkImpl(par3));
    return s;
}
} // namespace interface1
} // namespace training
} // namespace classification
} // namespace decision_forest
} // namespace algorithms
} // namespace daal
