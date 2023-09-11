/* file: stump_classification_modelimpl().cpp */
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
//  Implementation of the class defining the Decision tree model
//--
*/

#include "algorithms/decision_tree/decision_tree_classification_model.h"
#include "algorithms/stump/stump_classification_model.h"
#include "src/algorithms/classifier/classifier_model_impl.h"
#include "src/algorithms/decision_tree/decision_tree_classification_model_impl.h"
#include "src/services/service_defines.h"
#include "src/services/serialization_utils.h"
#include "src/services/daal_strings.h"
#include "src/algorithms/stump/stump_classification_model_visitor.h"

namespace daal
{
namespace algorithms
{
namespace stump
{
namespace classification
{
using namespace daal::data_management;
using namespace daal::services;

__DAAL_REGISTER_SERIALIZATION_CLASS(Model, SERIALIZATION_STUMP_CLASSIFICATION_MODEL_ID);

Model::Model() : _nClasses(0) {}

Model::~Model() {}

Model::Model(size_t nFeatures, size_t nClasses, services::Status & st) : decision_tree::classification::Model(nFeatures)
{
    if (!impl())
    {
        st.add(services::ErrorMemoryAllocationFailed);
    }
    _nClasses = nClasses;
}

services::SharedPtr<Model> Model::create(size_t nFeatures, size_t nClasses, services::Status * stat)
{
    DAAL_DEFAULT_CREATE_IMPL_EX(Model, nFeatures, nClasses);
}
services::Status Model::serializeImpl(data_management::InputDataArchive * arch)
{
    daal::algorithms::classifier::Model::serialImpl<data_management::InputDataArchive, false>(arch);
    impl()->serialImpl<data_management::InputDataArchive, false>(arch);

    return services::Status();
}

services::Status Model::deserializeImpl(const data_management::OutputDataArchive * arch)
{
    daal::algorithms::classifier::Model::serialImpl<const data_management::OutputDataArchive, true>(arch);
    impl()->serialImpl<const data_management::OutputDataArchive, true>(
        arch, COMPUTE_DAAL_VERSION(arch->getMajorVersion(), arch->getMinorVersion(), arch->getUpdateVersion()));

    return services::Status();
}

size_t Model::getSplitFeature() const
{
    StumpNodeVisitor visitor(_nClasses);
    traverseDFS(visitor);
    return visitor.splitFeature;
}

services::Status Parameter::check() const
{
    Status s;
    DAAL_CHECK_EX(nClasses >= 2, ErrorIncorrectParameter, ParameterName, nClassesStr());
    return s;
}

} // namespace classification
} // namespace stump
} // namespace algorithms
} // namespace daal
