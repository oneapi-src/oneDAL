/* file: stump_regression_modelimpl().cpp */
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

#include "algorithms/decision_tree/decision_tree_regression_model.h"
#include "algorithms/stump/stump_regression_model.h"
#include "src/algorithms/regression/regression_model_impl.h"
#include "src/algorithms/decision_tree/decision_tree_regression_model_impl.h"
#include "src/services/service_defines.h"
#include "src/services/serialization_utils.h"
#include "src/services/daal_strings.h"
#include "src/algorithms/stump/stump_regression_model_visitor.h"

namespace daal
{
namespace algorithms
{
namespace stump
{
namespace regression
{
using namespace daal::data_management;
using namespace daal::services;

__DAAL_REGISTER_SERIALIZATION_CLASS(Model, SERIALIZATION_STUMP_REGRESSION_MODEL_ID);

Model::~Model() {}

Model::Model(services::Status & st) : decision_tree::regression::Model()
{
    if (!impl())
    {
        st.add(services::ErrorMemoryAllocationFailed);
    }
}

services::SharedPtr<Model> Model::create(services::Status * stat)
{
    DAAL_DEFAULT_CREATE_IMPL(Model);
}
services::Status Model::serializeImpl(data_management::InputDataArchive * arch)
{
    daal::algorithms::regression::Model::serialImpl<data_management::InputDataArchive, false>(arch);
    impl()->serialImpl<data_management::InputDataArchive, false>(arch);

    return services::Status();
}

services::Status Model::deserializeImpl(const data_management::OutputDataArchive * arch)
{
    daal::algorithms::regression::Model::serialImpl<const data_management::OutputDataArchive, true>(arch);
    impl()->serialImpl<const data_management::OutputDataArchive, true>(
        arch, COMPUTE_DAAL_VERSION(arch->getMajorVersion(), arch->getMinorVersion(), arch->getUpdateVersion()));

    return services::Status();
}

size_t Model::getSplitFeature() const
{
    StumpNodeVisitor visitor;
    traverseDFS(visitor);
    return visitor.splitFeature;
}

services::Status Parameter::check() const
{
    services::Status s;
    return s;
}

} // namespace regression
} // namespace stump
} // namespace algorithms
} // namespace daal
