/* file: df_regression_training_input_v1.cpp */
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

#include "algorithms/decision_forest/decision_forest_regression_training_types.h"
#include "service/kernel/daal_strings.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace decision_forest
{
namespace training
{
Status checkImpl(const decision_forest::training::interface2::Parameter & prm);
}

namespace regression
{
namespace training
{
namespace interface1
{
Parameter::Parameter() {}

DAAL_FORCEINLINE void convertParameter(const interface1::Parameter & par1, interface2::Parameter & par2)
{
    par2.nTrees                      = par1.nTrees;
    par2.observationsPerTreeFraction = par1.observationsPerTreeFraction;
    par2.featuresPerNode             = par1.featuresPerNode;
    par2.maxTreeDepth                = par1.maxTreeDepth;
    par2.minObservationsInLeafNode   = par1.minObservationsInLeafNode;
    par2.seed                        = par1.seed;
    par2.engine                      = par1.engine;
    par2.impurityThreshold           = par1.impurityThreshold;
    par2.varImportance               = par1.varImportance;
    par2.resultsToCompute            = par1.resultsToCompute;
    par2.memorySavingMode            = par1.memorySavingMode;
    par2.bootstrap                   = par1.bootstrap;
}

Status Parameter::check() const
{
    decision_forest::regression::training::interface2::Parameter par2;
    convertParameter(*this, par2);
    return decision_forest::training::checkImpl(par2);
}
} // namespace interface1
} // namespace training
} // namespace regression
} // namespace decision_forest
} // namespace algorithms
} // namespace daal
