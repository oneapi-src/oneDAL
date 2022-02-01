/* file: ridge_regression_ne_model_fpt.cpp */
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
//  Implementation of the class defining the ridge regression model
//--
*/

#include "src/algorithms/ridge_regression/ridge_regression_ne_model_impl.h"
#include "data_management/data/homogen_numeric_table.h"

namespace daal
{
namespace algorithms
{
namespace ridge_regression
{
namespace internal
{
using namespace daal::data_management;

/**
 * Constructs the ridge regression model for the normal equations method
 * \param[in] featnum Number of features in the training data set
 * \param[in] nrhs    Number of responses in the training data
 * \param[in] par     Parameters of ridge regression model-based training
 * \param[in] dummy   Dummy variable for the templated constructor
 */
template <typename modelFPType>
ModelNormEqInternal::ModelNormEqInternal(size_t featnum, size_t nrhs, const ridge_regression::Parameter & par, modelFPType dummy, Status & st)
    : super(featnum, nrhs, par, dummy)
{
    size_t dimWithoutBeta = getNumberOfBetas();
    if (!_interceptFlag)
    {
        dimWithoutBeta--;
    }

    _xtxTable = HomogenNumericTable<modelFPType>::create(dimWithoutBeta, dimWithoutBeta, NumericTable::doAllocate, 0, &st);
    if (!st) return;
    _xtyTable = HomogenNumericTable<modelFPType>::create(dimWithoutBeta, nrhs, NumericTable::doAllocate, 0, &st);
    if (!st) return;
}

template ModelNormEqInternal::ModelNormEqInternal(size_t featnum, size_t nrhs, const ridge_regression::Parameter & par, DAAL_FPTYPE dummy,
                                                  Status & st);
} // namespace internal
} // namespace ridge_regression
} // namespace algorithms
} // namespace daal
