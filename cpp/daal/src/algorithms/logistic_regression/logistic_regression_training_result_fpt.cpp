/* file: logistic_regression_training_result_fpt.cpp */
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
//  Implementation of the logistic regression algorithm interface
//--
*/

#include "algorithms/logistic_regression/logistic_regression_training_types.h"
#include "src/algorithms/logistic_regression/logistic_regression_model_impl.h"
#include "data_management/data/homogen_numeric_table.h"

namespace daal
{
namespace algorithms
{
namespace logistic_regression
{
namespace internal
{

template <typename modelFPType>
ModelImpl::ModelImpl(size_t nFeatures, bool interceptFlag, size_t nClasses, modelFPType dummy, services::Status * st)
    : ClassificationImplType(nFeatures), _interceptFlag(interceptFlag)
{
    const size_t nRows = nClasses == 2 ? 1 : nClasses;
    const size_t nCols = nFeatures + 1;

    _beta = data_management::HomogenNumericTable<modelFPType>::create(nCols, nRows, data_management::NumericTable::doAllocate, 0, st);
}

template ModelImpl::ModelImpl(size_t nFeatures, bool interceptFlag, size_t nClasses, DAAL_FPTYPE dummy, services::Status * st);

} // namespace internal

namespace training
{
namespace interface2
{
template <typename algorithmFPType>
DAAL_EXPORT services::Status Result::allocate(const daal::algorithms::Input * input, const daal::algorithms::Parameter * parameter, const int method)
{
    services::Status s;
    const classifier::training::Input * inp              = static_cast<const classifier::training::Input *>(input);
    const size_t nFeatures                               = inp->get(classifier::training::data)->getNumberOfColumns();
    const logistic_regression::training::Parameter * prm = (const logistic_regression::training::Parameter *)parameter;
    set(classifier::training::model,
        ModelPtr(new logistic_regression::internal::ModelImpl(nFeatures, prm->interceptFlag, prm->nClasses, algorithmFPType(0), &s)));
    return s;
}

template DAAL_EXPORT services::Status Result::allocate<DAAL_FPTYPE>(const daal::algorithms::Input * input,
                                                                    const daal::algorithms::Parameter * parameter, const int method);
} // namespace interface2

} // namespace training
} // namespace logistic_regression
} // namespace algorithms
} // namespace daal
