/* file: stump_model.cpp */
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

/*
//++
//  Implementation of the decision stump model constructor.
//--
*/

#include "algorithms/stump/stump_model.h"

namespace daal
{
namespace algorithms
{
namespace stump
{
/**
 * Empty constructor for deserialization
 */
Model::Model() : weak_learner::Model(), _nFeatures(0), _splitFeature(0), _values() {}

size_t Model::getSplitFeature()
{
    return _splitFeature;
}

void Model::setSplitFeature(size_t splitFeature)
{
    _splitFeature = splitFeature;
}

} // namespace stump
} // namespace algorithms
} // namespace daal
