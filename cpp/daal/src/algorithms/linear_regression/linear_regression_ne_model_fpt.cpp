/* file: linear_regression_ne_model_fpt.cpp */
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
//  Implementation of the class defining the linear regression model
//--
*/

#include "src/algorithms/linear_regression/linear_regression_ne_model_impl.h"
#include "data_management/data/homogen_numeric_table.h"
#include "services/internal/execution_context.h"

namespace daal
{
namespace algorithms
{
namespace linear_regression
{
namespace internal
{
using namespace daal::data_management;

/**
 * Constructs the linear regression model for the normal equations method
 * \param[in] featnum Number of features in the training data set
 * \param[in] nrhs    Number of responses in the training data
 * \param[in] par     Parameters of linear regression model-based training
 * \param[in] dummy   Dummy variable for the templated constructor
 */
template <typename modelFPType>
ModelNormEqInternal::ModelNormEqInternal(size_t featnum, size_t nrhs, const linear_regression::Parameter & par, modelFPType dummy, Status & st)
    : super(featnum, nrhs, par, dummy)
{
    size_t dimWithoutBeta = getNumberOfBetas();
    if (!_interceptFlag)
    {
        dimWithoutBeta--;
    }

    auto & context    = services::internal::getDefaultContext();
    auto & deviceInfo = context.getInfoDevice();

    if (deviceInfo.isCpu)
    {
        _xtxTable = HomogenNumericTable<modelFPType>::create(dimWithoutBeta, dimWithoutBeta, NumericTable::doAllocate, 0, &st);
        if (!st) return;
        _xtyTable = HomogenNumericTable<modelFPType>::create(dimWithoutBeta, nrhs, NumericTable::doAllocate, 0, &st);
        if (!st) return;
    }
}

template ModelNormEqInternal::ModelNormEqInternal(size_t featnum, size_t nrhs, const linear_regression::Parameter & par, DAAL_FPTYPE dummy,
                                                  Status & st);
} // namespace internal
} // namespace linear_regression
} // namespace algorithms
} // namespace daal
