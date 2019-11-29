/* file: svm_model_fpt.cpp */
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
//  Implementation of the class defining the SVM model.
//--
*/

#include "algorithms/svm/svm_model.h"

namespace daal
{
namespace algorithms
{
namespace svm
{
namespace interface1
{
template <typename modelFPType>
services::SharedPtr<Model> Model::create(size_t nColumns, data_management::NumericTableIface::StorageLayout layout, services::Status * stat)
{
    DAAL_DEFAULT_CREATE_IMPL_EX(Model, (modelFPType)0.0, nColumns, layout);
}

template <typename modelFPType>
Model::Model(modelFPType dummy, size_t nColumns, data_management::NumericTableIface::StorageLayout layout, services::Status & st) : _bias(0.0)
{
    using namespace data_management;
    if (layout == NumericTableIface::csrArray)
    {
        _SV = CSRNumericTable::create<modelFPType>(NULL, NULL, NULL, nColumns, 0, CSRNumericTable::oneBased, &st);
    }
    else
    {
        _SV = HomogenNumericTable<modelFPType>::create(NULL, nColumns, 0, &st);
    }
    if (!st) return;
    _SVCoeff = HomogenNumericTable<modelFPType>::create(NULL, 1, 0, &st);
    if (!st) return;
    _SVIndices = HomogenNumericTable<int>::create(NULL, 1, 0, &st);
    if (!st) return;
}

template DAAL_EXPORT services::SharedPtr<Model> Model::create<DAAL_FPTYPE>(size_t nColumns, data_management::NumericTableIface::StorageLayout layout,
                                                                           services::Status * stat);

template DAAL_EXPORT Model::Model(DAAL_FPTYPE, size_t, data_management::NumericTableIface::StorageLayout, services::Status &);

} // namespace interface1
} // namespace svm
} // namespace algorithms
} // namespace daal
