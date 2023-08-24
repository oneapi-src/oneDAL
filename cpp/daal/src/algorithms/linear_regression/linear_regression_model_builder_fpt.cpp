/* file: linear_regression_model_builder_fpt.cpp */
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

#include "algorithms/linear_regression/linear_regression_model_builder.h"
#include "src/algorithms/linear_regression/linear_regression_model_impl.h"
#include "data_management/data/homogen_numeric_table.h"

namespace daal
{
namespace algorithms
{
namespace linear_regression
{
using namespace daal::data_management;

template <typename modelFPType>
ModelBuilder<modelFPType>::ModelBuilder()
    : _nFeatures(0), _nResponses(0), _modelPtr(linear_regression::ModelPtr(new linear_regression::internal::ModelImpl()))
{}

template <typename modelFPType>
ModelBuilder<modelFPType>::ModelBuilder(size_t nFeatures, size_t nResponses) : _nFeatures(nFeatures), _nResponses(nResponses)
{
    Parameter * p     = new Parameter;
    p->interceptFlag  = true; /* default intercept flag is true but can be changed via setInterceptFlag */
    modelFPType dummy = 1.0;
    _modelPtr         = linear_regression::ModelPtr(new linear_regression::internal::ModelImpl(_nFeatures, _nResponses, *p, dummy));
}

template <typename modelFPType>
void ModelBuilder<modelFPType>::setInterceptFlag(bool interceptFlag)
{
    linear_regression::internal::ModelImpl * m = static_cast<linear_regression::internal::ModelImpl *>(_modelPtr.get());
    m->setInterceptFlag(interceptFlag);
}

template class ModelBuilder<DAAL_FPTYPE>;
} // namespace linear_regression
} // namespace algorithms
} // namespace daal
