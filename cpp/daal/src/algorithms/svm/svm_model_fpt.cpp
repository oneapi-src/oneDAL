/* file: svm_model_fpt.cpp */
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
//  Implementation of the class defining the SVM model.
//--
*/

#include "algorithms/svm/svm_model.h"
#include "data_management/data/homogen_numeric_table.h"
#include "data_management/data/csr_numeric_table.h"

namespace daal
{
namespace algorithms
{
namespace svm
{
namespace interface1
{
namespace dm  = daal::data_management;
namespace dmi = daal::data_management::internal;
template <typename modelFPType>
services::SharedPtr<Model> Model::create(size_t nColumns, data_management::NumericTableIface::StorageLayout layout, services::Status * stat)
{
    DAAL_DEFAULT_CREATE_IMPL_EX(Model, (modelFPType)0.0, nColumns, layout);
}

template <typename modelFPType>
Model::Model(modelFPType dummy, size_t nColumns, data_management::NumericTableIface::StorageLayout layout, services::Status & st) : _bias(0.0)
{
    if (layout == dm::NumericTableIface::csrArray)
    {
        _SV = dm::CSRNumericTable::create<modelFPType>(NULL, NULL, NULL, nColumns, 0, dm::CSRNumericTable::oneBased, &st);
    }
    else
    {
        _SV = dm::HomogenNumericTable<modelFPType>::create(NULL, nColumns, 0, &st);
    }
    _SVCoeff = dm::HomogenNumericTable<modelFPType>::create(NULL, 1, 0, &st);
    if (!st) return;
    _SVIndices = dm::HomogenNumericTable<int>::create(NULL, 1, 0, &st);

    return;
}

template DAAL_EXPORT services::SharedPtr<Model> Model::create<DAAL_FPTYPE>(size_t nColumns, data_management::NumericTableIface::StorageLayout layout,
                                                                           services::Status * stat);

template DAAL_EXPORT Model::Model(DAAL_FPTYPE, size_t, dm::NumericTableIface::StorageLayout, services::Status &);

} // namespace interface1
} // namespace svm
} // namespace algorithms
} // namespace daal
