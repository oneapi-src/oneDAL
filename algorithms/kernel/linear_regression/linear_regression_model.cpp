/* file: linear_regression_model.cpp */
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

Status checkModel(
    linear_regression::Model* model, const daal::algorithms::Parameter &par, size_t nBeta, size_t nResponses, int method)
{
    services::Status s;
    DAAL_CHECK_STATUS(s, linear_model::checkModel(model, par, nBeta, nResponses));

    size_t dimWithoutBeta = (model->getInterceptFlag() ? nBeta : nBeta - 1);

    if(method == linear_regression::training::normEqDense)
    {
        linear_regression::ModelNormEq* modelNormEq = dynamic_cast<linear_regression::ModelNormEq*>(model);
        DAAL_CHECK(modelNormEq, ErrorIncorrectTypeOfModel);

        DAAL_CHECK_STATUS(s, checkNumericTable(modelNormEq->getXTXTable().get(), XTXTableStr(), 0, 0, dimWithoutBeta, dimWithoutBeta));
        DAAL_CHECK_STATUS(s, checkNumericTable(modelNormEq->getXTYTable().get(), XTYTableStr(), 0, 0, dimWithoutBeta, nResponses));
    }
    else if(method == linear_regression::training::qrDense)
    {
        linear_regression::ModelQR* modelQR = dynamic_cast<linear_regression::ModelQR*>(model);
        DAAL_CHECK(modelQR, ErrorIncorrectTypeOfModel);

        DAAL_CHECK_STATUS(s, checkNumericTable(modelQR->getRTable().get(), RTableStr(), 0, 0, dimWithoutBeta, dimWithoutBeta));
        DAAL_CHECK_STATUS(s, checkNumericTable(modelQR->getQTYTable().get(), QTYTableStr(), 0, 0, dimWithoutBeta, nResponses));
    }

    return s;
}

} // namespace linear_regression
} // namespace algorithms
} // namespace daal
