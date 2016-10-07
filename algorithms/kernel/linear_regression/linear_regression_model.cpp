/* file: linear_regression_model.cpp */
/*******************************************************************************
* Copyright 2014-2016 Intel Corporation
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
#include "algorithms/linear_regression/linear_regression_types.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace linear_regression
{
namespace interface1
{

Parameter::Parameter() : interceptFlag(true) {}

/**
 * Constructs the linear regression model
 * \param[in] beta  Numeric table that contains linear regression coefficients
 * \param[in] par   Linear regression parameters
 */
Model::Model(NumericTablePtr &beta, const Parameter &par) : daal::algorithms::Model()
{
    _coefdim = beta->getNumberOfColumns();
    _nrhs = beta->getNumberOfRows();
    _interceptFlag = par.interceptFlag;
    _beta = beta;
}

void Model::initialize()
{
    BlockDescriptor<double> _betaArrayBlock;
    double *_betaArray = NULL;
    _beta->getBlockOfRows(0, _nrhs, writeOnly, _betaArrayBlock);
    _betaArray = _betaArrayBlock.getBlockPtr();

    for(size_t i = 0; i < _coefdim * _nrhs; i++)
    {
        _betaArray[i] = 0;
    }

    _beta->releaseBlockOfRows(_betaArrayBlock);
}

/**
 * Returns the number of regression coefficients
 * \return Number of regression coefficients
 */
size_t Model::getNumberOfBetas() { return _coefdim; }

/**
 * Returns the number of features in the training data set
 * \return Number of features in the training data set
 */
size_t Model::getNumberOfFeatures() { return _coefdim - 1; }

/**
 * Returns the number of responses in the training data set
 * \return Number of responses in the training data set
 */
size_t Model::getNumberOfResponses() { return _nrhs; }

/**
 * Returns true if the linear regression model contains the intercept term, and false otherwise
 * \return True if the linear regression model contains the intercept term, and false otherwise
 */
bool Model::getInterceptFlag() { return _interceptFlag; }

/**
 * Returns the numeric table that contains regression coefficients
 * \return Table that contains regression coefficients
 */
NumericTablePtr Model::getBeta() { return _beta; }

void Model::setToZero(NumericTable *table)
{
    BlockDescriptor<double> block;
    double *tableArray;

    size_t nRows = table->getNumberOfRows();
    size_t nCols = table->getNumberOfColumns();

    table->getBlockOfRows(0, nRows, writeOnly, block);
    tableArray = block.getBlockPtr();

    for(size_t i = 0; i < nCols * nRows; i++)
    {
        tableArray[i] = 0;
    }
    table->releaseBlockOfRows(block);
}

} // namespace interface1
} // namespace linear_regression
} // namespace algorithms
} // namespace daal


/**
 * Checks the correctness of linear regression model
 * \param[in]  model             The model to check
 * \param[in]  par               The parameter of linear regression algorithm
 * \param[out] errors            The collection of errors
 * \param[in]  coefdim           Required number of linear regression coefficients
 * \param[in]  nrhs              Required number of responses on the training stage
 * \param[in]  method            Computation method
 */
void daal::algorithms::linear_regression::checkModel(linear_regression::Model* model, const daal::algorithms::Parameter *par, services::ErrorCollection *errors,
    const size_t coefdim, const size_t nrhs, int method)
{
    if(!model) { errors->add(ErrorNullModel); return; }

    const Parameter *parameter = static_cast<const Parameter *>(par);

    if(model->getInterceptFlag() != parameter->interceptFlag)
    {
        errors->add(services::Error::create(ErrorIncorrectParameter, ParameterName, interceptFlagStr()));
        return;
    }

    if(!checkNumericTable(model->getBeta().get(), errors, betaStr(), 0, 0, coefdim, nrhs)) { return; }

    size_t dimWithoutBeta = coefdim;
    if(!model->getInterceptFlag())
    {
        dimWithoutBeta--;
    }

    if(method == linear_regression::training::normEqDense)
    {
        linear_regression::ModelNormEq* modelNormEq = dynamic_cast<linear_regression::ModelNormEq*>(model);
        if(!modelNormEq) { errors->add(ErrorIncorrectTypeOfModel); return; }

        if(!checkNumericTable(modelNormEq->getXTXTable().get(), errors, XTXTableStr(), 0, 0, dimWithoutBeta, dimWithoutBeta)) { return; }
        if(!checkNumericTable(modelNormEq->getXTYTable().get(), errors, XTYTableStr(), 0, 0, dimWithoutBeta, nrhs)) { return; }
    }
    else if(method == linear_regression::training::qrDense)
    {
        linear_regression::ModelQR* modelQR = dynamic_cast<linear_regression::ModelQR*>(model);
        if(!modelQR) { errors->add(ErrorIncorrectTypeOfModel); return; }

        if(!checkNumericTable(modelQR->getRTable().get(), errors, RTableStr(), 0, 0, dimWithoutBeta, dimWithoutBeta)) { return; }
        if(!checkNumericTable(modelQR->getQTYTable().get(), errors, QTYTableStr(), 0, 0, dimWithoutBeta, nrhs)) { return; }
    }
}
