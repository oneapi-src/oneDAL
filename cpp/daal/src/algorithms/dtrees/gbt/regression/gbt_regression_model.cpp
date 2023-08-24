/* file: gbt_regression_model.cpp */
/*******************************************************************************
* Copyright 2014 Intel Corporation
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
//  Implementation of the class defining the gradient boosted trees model
//--
*/

#include "algorithms/gradient_boosted_trees/gbt_regression_model.h"
#include "src/services/serialization_utils.h"
#include "src/algorithms/dtrees/gbt/regression/gbt_regression_model_impl.h"

using namespace daal::data_management;
using namespace daal::services;
using namespace daal::algorithms::gbt::internal;

namespace daal
{
namespace algorithms
{
namespace gbt
{
using namespace daal::algorithms::gbt::internal;

namespace regression
{
__DAAL_REGISTER_SERIALIZATION_CLASS2(Model, internal::ModelImpl, SERIALIZATION_GBT_REGRESSION_MODEL_ID);
Model::Model() {}

ModelPtr Model::create(size_t nFeatures, services::Status * stat)
{
    daal::algorithms::gbt::regression::ModelPtr pRes(new gbt::regression::internal::ModelImpl(nFeatures));
    if ((!pRes.get()) && stat) stat->add(services::ErrorMemoryAllocationFailed);
    return pRes;
}

namespace internal
{
size_t ModelImpl::numberOfTrees() const
{
    return ImplType::numberOfTrees();
}

size_t ModelImpl::getNumberOfTrees() const
{
    return ImplType::numberOfTrees();
}

void ModelImpl::traverseDF(size_t iTree, algorithms::regression::TreeNodeVisitor & visitor) const
{
    ImplType::traverseDF(iTree, visitor);
}

void ModelImpl::traverseBF(size_t iTree, algorithms::regression::TreeNodeVisitor & visitor) const
{
    ImplType::traverseBF(iTree, visitor);
}

void ModelImpl::traverseDFS(size_t iTree, tree_utils::regression::TreeNodeVisitor & visitor) const
{
    ImplType::traverseDFS(iTree, visitor);
}

void ModelImpl::traverseBFS(size_t iTree, tree_utils::regression::TreeNodeVisitor & visitor) const
{
    ImplType::traverseBFS(iTree, visitor);
}

services::Status ModelImpl::serializeImpl(data_management::InputDataArchive * arch)
{
    auto s = algorithms::regression::Model::serialImpl<data_management::InputDataArchive, false>(arch);
    s.add(algorithms::regression::internal::ModelInternal::serialImpl<data_management::InputDataArchive, false>(arch));
    return s.add(ImplType::serialImpl<data_management::InputDataArchive, false>(arch));
}

services::Status ModelImpl::deserializeImpl(const data_management::OutputDataArchive * arch)
{
    auto s = algorithms::regression::Model::serialImpl<const data_management::OutputDataArchive, true>(arch);
    s.add(algorithms::regression::internal::ModelInternal::serialImpl<const data_management::OutputDataArchive, true>(arch));
    return s.add(ImplType::serialImpl<const data_management::OutputDataArchive, true>(arch));
}

} // namespace internal
} // namespace regression
} // namespace gbt
} // namespace algorithms
} // namespace daal
