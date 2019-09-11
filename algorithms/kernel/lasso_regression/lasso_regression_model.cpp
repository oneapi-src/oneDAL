/* file: lasso_regression_model.cpp */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation.
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
//  Implementation of the class defining the lasso regression model
//--
*/

#include "lasso_regression_model_impl.h"
#include "serialization_utils.h"
#include "service_numeric_table.h"

using namespace daal::data_management;
using namespace daal::services;
using namespace daal::algorithms::lasso_regression::internal;

namespace daal
{
namespace algorithms
{
namespace lasso_regression
{
namespace interface1
{
__DAAL_REGISTER_SERIALIZATION_CLASS2(Model, internal::ModelImpl, SERIALIZATION_LASSO_REGRESSION_MODEL_ID);
}
services::Status checkModel(lasso_regression::Model* model, const daal::algorithms::Parameter &par, size_t nBeta, size_t nResponses, int method)
{
    services::Status s;
    DAAL_CHECK_STATUS(s, linear_model::checkModel(model, par, nBeta, nResponses));

    return s;
}

} // namespace lasso_regression
} // namespace algorithms
} // namespace daal
