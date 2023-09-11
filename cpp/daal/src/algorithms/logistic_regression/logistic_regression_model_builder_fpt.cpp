/* file: logistic_regression_model_builder_fpt.cpp */
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

#include "algorithms/logistic_regression/logistic_regression_model_builder.h"
#include "src/algorithms/logistic_regression/logistic_regression_model_impl.h"
#include "data_management/data/homogen_numeric_table.h"

namespace daal
{
namespace algorithms
{
namespace logistic_regression
{
using namespace daal::data_management;

template <typename modelFPType>
ModelBuilder<modelFPType>::ModelBuilder()
    : _nFeatures(0), _nClasses(0), _modelPtr(logistic_regression::ModelPtr(new logistic_regression::internal::ModelImpl()))
{}

template <typename modelFPType>
ModelBuilder<modelFPType>::ModelBuilder(size_t nFeatures, size_t nClasses) : _nFeatures(nFeatures), _nClasses(nClasses)
{
    const bool interceptFlag = true; /* default intercept flag is true but can be changed via setInterceptFlag */
    modelFPType dummy        = 1.0;
    _modelPtr = logistic_regression::ModelPtr(new logistic_regression::internal::ModelImpl(_nFeatures, interceptFlag, _nClasses, dummy, &_s));
}

template <typename modelFPType>
void ModelBuilder<modelFPType>::setInterceptFlag(bool interceptFlag)
{
    logistic_regression::internal::ModelImpl * const m = static_cast<logistic_regression::internal::ModelImpl *>(_modelPtr.get());
    _s                                                 = m->reset(interceptFlag);
}

template class ModelBuilder<DAAL_FPTYPE>;
} // namespace logistic_regression
} // namespace algorithms
} // namespace daal
