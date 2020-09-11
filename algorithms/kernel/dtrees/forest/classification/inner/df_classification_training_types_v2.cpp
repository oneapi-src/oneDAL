/* file: df_classification_training_types_v2.cpp */
/*******************************************************************************
* Copyright 2014-2020 Intel Corporation
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

#include "algorithms/kernel/dtrees/forest/classification/df_classification_training_types_result.h"
#include "service/kernel/serialization_utils.h"
#include "service/kernel/daal_strings.h"

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
namespace interface2
{
DAAL_FORCEINLINE void convertParameter(const interface2::Parameter & par2, interface3::Parameter & par3)
{
    par3.nTrees                      = par2.nTrees;
    par3.observationsPerTreeFraction = par2.observationsPerTreeFraction;
    par3.featuresPerNode             = par2.featuresPerNode;
    par3.maxTreeDepth                = par2.maxTreeDepth;
    par3.minObservationsInLeafNode   = par2.minObservationsInLeafNode;
    par3.seed                        = par2.seed;
    par3.engine                      = par2.engine;
    par3.impurityThreshold           = par2.impurityThreshold;
    par3.varImportance               = par2.varImportance;
    par3.resultsToCompute            = par2.resultsToCompute;
    par3.memorySavingMode            = par2.memorySavingMode;
    par3.bootstrap                   = par2.bootstrap;
}

services::Status Parameter::check() const
{
    services::Status s;
    DAAL_CHECK_STATUS(s, classifier::Parameter::check());

    decision_forest::classification::training::interface3::Parameter par3(this->nClasses);
    convertParameter(*this, par3);

    DAAL_CHECK_STATUS(s, decision_forest::training::checkImpl(par3));
    return s;
}
} // namespace interface2
} // namespace training
} // namespace classification
} // namespace decision_forest
} // namespace algorithms
} // namespace daal
