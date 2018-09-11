/* file: gbt_classification_model.cpp */
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

/*
//++
//  Implementation of the class defining the gradient boosted trees model
//--
*/

#include "algorithms/gradient_boosted_trees/gbt_classification_model.h"
#include "serialization_utils.h"
#include "gbt_classification_model_impl.h"

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

namespace classification
{
namespace interface1
{
__DAAL_REGISTER_SERIALIZATION_CLASS2(Model, internal::ModelImpl, SERIALIZATION_GBT_CLASSIFICATION_MODEL_ID);

ModelPtr Model::create(size_t nFeatures, services::Status *stat)
{
    daal::algorithms::gbt::classification::ModelPtr pRes(new gbt::classification::internal::ModelImpl(nFeatures));
    if((!pRes.get()) && stat)
        stat->add(services::ErrorMemoryAllocationFailed);
    return pRes;
}

}

namespace internal
{

size_t ModelImpl::numberOfTrees() const
{
    return ImplType::numberOfTrees();
}

void ModelImpl::traverseDF(size_t iTree, algorithms::regression::TreeNodeVisitor& visitor) const
{
    ImplType::traverseDF(iTree, visitor);
}

void ModelImpl::traverseBF(size_t iTree, algorithms::regression::TreeNodeVisitor& visitor) const
{
    ImplType::traverseBF(iTree, visitor);
}

void ModelImpl::traverseDFS(size_t iTree, tree_utils::regression::TreeNodeVisitor& visitor) const
{
    ImplType::traverseDFS(iTree, visitor);
}

void ModelImpl::traverseBFS(size_t iTree, tree_utils::regression::TreeNodeVisitor& visitor) const
{
    ImplType::traverseBFS(iTree, visitor);
}

services::Status ModelImpl::serializeImpl(data_management::InputDataArchive  * arch)
{
    auto s = algorithms::classifier::Model::serialImpl<data_management::InputDataArchive, false>(arch);
    arch->set(this->_nFeatures); //algorithms::classifier::internal::ModelInternal
    return s.add(ImplType::serialImpl<data_management::InputDataArchive, false>(arch));
}

services::Status ModelImpl::deserializeImpl(const data_management::OutputDataArchive * arch)
{
    auto s = algorithms::classifier::Model::serialImpl<const data_management::OutputDataArchive, true>(arch);
    arch->set(this->_nFeatures); //algorithms::classifier::internal::ModelInternal
    return s.add(ImplType::serialImpl<const data_management::OutputDataArchive, true>(arch,
        COMPUTE_DAAL_VERSION(arch->getMajorVersion(), arch->getMinorVersion(), arch->getUpdateVersion())));
}

} // namespace interface1
} // namespace classification
} // namespace gbt
} // namespace algorithms
} // namespace daal
