/* file: ridge_regression_model.cpp */
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
//  Implementation of the class defining the ridge regression model
//--
*/

#include "algorithms/ridge_regression/ridge_regression_model.h"
#include "algorithms/ridge_regression/ridge_regression_ne_model.h"
#include "data_management/data/homogen_numeric_table.h"
#include "daal_strings.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace ridge_regression
{
namespace interface1
{

TrainParameter::TrainParameter()
        : Parameter(),
          ridgeParameters(new HomogenNumericTable<double>(1, 1, NumericTableIface::doAllocate, 1.0))
    {
    };

services::Status TrainParameter::check() const
{
    return checkNumericTable(ridgeParameters.get(), ridgeParametersStr(), packed_mask, 0, 0, 1);
}

} // namespace interface1

Status checkModel(
    ridge_regression::Model* model, const daal::algorithms::Parameter &par, size_t nBeta, size_t nResponses, int method)
{
    services::Status s;
    DAAL_CHECK_STATUS(s, linear_model::checkModel(model, par, nBeta, nResponses));

    size_t dimWithoutBeta = (model->getInterceptFlag() ? nBeta : nBeta - 1);

    ridge_regression::ModelNormEq* modelNormEq = dynamic_cast<ridge_regression::ModelNormEq*>(model);

    DAAL_CHECK_STATUS(s, checkNumericTable(modelNormEq->getXTXTable().get(), XTXTableStr(), 0, 0, dimWithoutBeta, dimWithoutBeta));
    return               checkNumericTable(modelNormEq->getXTYTable().get(), XTYTableStr(), 0, 0, dimWithoutBeta, nResponses);
}
} // namespace ridge_regression
} // namespace algorithms
} // namespace daal
