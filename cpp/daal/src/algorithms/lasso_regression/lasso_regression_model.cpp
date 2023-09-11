/* file: lasso_regression_model.cpp */
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
//  Implementation of the class defining the lasso regression model
//--
*/

#include "src/algorithms/lasso_regression/lasso_regression_model_impl.h"
#include "src/services/serialization_utils.h"
#include "src/data_management/service_numeric_table.h"

using namespace daal::data_management;
using namespace daal::services;
using namespace daal::algorithms::lasso_regression::internal;

namespace daal
{
namespace algorithms
{
namespace lasso_regression
{
__DAAL_REGISTER_SERIALIZATION_CLASS2(Model, internal::ModelImpl, SERIALIZATION_LASSO_REGRESSION_MODEL_ID);
services::Status checkModel(lasso_regression::Model * model, const daal::algorithms::Parameter & par, size_t nBeta, size_t nResponses, int method)
{
    services::Status s;
    DAAL_CHECK_STATUS(s, linear_model::checkModel(model, par, nBeta, nResponses));

    return s;
}

} // namespace lasso_regression
} // namespace algorithms
} // namespace daal
