/* file: linear_regression_model.cpp */
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
//  Implementation of the class defining the linear regression model
//--
*/

#include "algorithms/linear_regression/linear_regression_model.h"
#include "algorithms/linear_regression/linear_regression_ne_model.h"
#include "algorithms/linear_regression/linear_regression_qr_model.h"
#include "algorithms/linear_regression/linear_regression_training_types.h"
#include "daal_strings.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace linear_regression
{
Status checkModel(linear_regression::Model * model, const daal::algorithms::Parameter & par, size_t nBeta, size_t nResponses, int method)
{
    services::Status s;
    DAAL_CHECK_STATUS(s, linear_model::checkModel(model, par, nBeta, nResponses));

    size_t dimWithoutBeta = (model->getInterceptFlag() ? nBeta : nBeta - 1);

    if (method == linear_regression::training::normEqDense)
    {
        linear_regression::ModelNormEq * modelNormEq = dynamic_cast<linear_regression::ModelNormEq *>(model);
        DAAL_CHECK(modelNormEq, ErrorIncorrectTypeOfModel);

        DAAL_CHECK_STATUS(s, checkNumericTable(modelNormEq->getXTXTable().get(), XTXTableStr(), 0, 0, dimWithoutBeta, dimWithoutBeta));
        DAAL_CHECK_STATUS(s, checkNumericTable(modelNormEq->getXTYTable().get(), XTYTableStr(), 0, 0, dimWithoutBeta, nResponses));
    }
    else if (method == linear_regression::training::qrDense)
    {
        linear_regression::ModelQR * modelQR = dynamic_cast<linear_regression::ModelQR *>(model);
        DAAL_CHECK(modelQR, ErrorIncorrectTypeOfModel);

        DAAL_CHECK_STATUS(s, checkNumericTable(modelQR->getRTable().get(), RTableStr(), 0, 0, dimWithoutBeta, dimWithoutBeta));
        DAAL_CHECK_STATUS(s, checkNumericTable(modelQR->getQTYTable().get(), QTYTableStr(), 0, 0, dimWithoutBeta, nResponses));
    }

    return s;
}

} // namespace linear_regression
} // namespace algorithms
} // namespace daal
