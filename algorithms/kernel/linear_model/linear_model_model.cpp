/* file: linear_model_model.cpp */
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
//  Implementation of the class defining the regression model
//--
*/

#include "linear_model_model_impl.h"
#include "algorithms/linear_model/linear_model_model.h"
#include "service_numeric_table.h"
#include "daal_strings.h"

namespace daal
{
namespace algorithms
{
namespace linear_model
{
using namespace daal::services;
using namespace daal::data_management;

namespace interface1
{
Parameter::Parameter() : algorithms::Parameter(), interceptFlag(true) {}
Parameter::Parameter(const Parameter &other) : algorithms::Parameter(other), interceptFlag(other.interceptFlag) {}
}

namespace internal
{
ModelInternal::ModelInternal() : _interceptFlag(true), _beta() {}

ModelInternal::ModelInternal(const NumericTablePtr &beta, const linear_model::Parameter &par) :
    _beta(beta), _interceptFlag(par.interceptFlag)
{}

Status ModelInternal::initialize()
{
    const size_t nRows = _beta->getNumberOfRows();
    daal::internal::WriteOnlyRows<float, sse2> betaRows(*_beta, 0, nRows);
    DAAL_CHECK_BLOCK_STATUS(betaRows);
    float *betaArray = betaRows.get();
    const size_t betaSize = _beta->getNumberOfColumns() * nRows;
    for(size_t i = 0; i < betaSize; i++)
    {
        betaArray[i] = 0.0f;
    }
    return Status();
}

size_t ModelInternal::getNumberOfBetas() const { return _beta->getNumberOfColumns(); }

size_t ModelInternal::getNumberOfFeatures() const { return getNumberOfBetas() - 1; }

size_t ModelInternal::getNumberOfResponses() const { return _beta->getNumberOfRows(); }

bool ModelInternal::getInterceptFlag() const { return _interceptFlag; }

NumericTablePtr ModelInternal::getBeta() { return _beta; }

Status ModelInternal::setToZero(NumericTable &table)
{
    const size_t nRows = table.getNumberOfRows();
    daal::internal::WriteOnlyRows<float, sse2> tableRows(table, 0, nRows);
    DAAL_CHECK_BLOCK_STATUS(tableRows);
    float *tableArray = tableRows.get();

    const size_t nCols = table.getNumberOfColumns();

    for(size_t i = 0; i < nCols * nRows; i++)
    {
        tableArray[i] = 0.0f;
    }

    return Status();
}

} // namespace internal

Status checkModel(linear_model::Model* model, const daal::algorithms::Parameter &par, size_t nBeta, size_t nResponses)
{
    DAAL_CHECK(model, ErrorNullModel);

    const Parameter &parameter = static_cast<const Parameter &>(par);
    DAAL_CHECK_EX(model->getInterceptFlag() == parameter.interceptFlag, ErrorIncorrectParameter, ParameterName, interceptFlagStr());

    return checkNumericTable(model->getBeta().get(), betaStr(), 0, 0, nBeta, nResponses);
}
}
}
}
