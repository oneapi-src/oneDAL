/* file: kdtree_knn_classification_model_impl.cpp */
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

/*
//++
//  Implementation of the class defining the K-Nearest Neighbors (kNN) model
//--
*/

#include "kdtree_knn_classification_model_impl.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace kdtree_knn_classification
{
namespace interface1
{

Model::Model() : daal::algorithms::classifier::Model(), _impl(new ModelImpl) {}

Model::~Model()
{
    delete _impl;
}

void Model::serializeImpl(data_management::InputDataArchive  * arch)
{
    daal::algorithms::Model::serialImpl<data_management::InputDataArchive, false>(arch);
    _impl->serialImpl<data_management::InputDataArchive, false>(arch);
}

void Model::deserializeImpl(data_management::OutputDataArchive * arch)
{
    daal::algorithms::Model::serialImpl<data_management::OutputDataArchive, true>(arch);
    _impl->serialImpl<data_management::OutputDataArchive, true>(arch);
}

} // namespace interface1
} // namespace kdtree_knn_classification
} // namespace algorithms
} // namespace daal
