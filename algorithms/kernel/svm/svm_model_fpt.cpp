/* file: svm_model_fpt.cpp */
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

template<typename modelFPType>
services::SharedPtr<Model> Model::create(size_t nColumns,
                                         data_management::NumericTableIface::StorageLayout layout,
                                         services::Status *stat)
{
    DAAL_DEFAULT_CREATE_IMPL_EX(Model, (modelFPType)0.0, nColumns, layout);
}

template<typename modelFPType>
Model::Model(modelFPType dummy, size_t nColumns, data_management::NumericTableIface::StorageLayout layout,
      services::Status &st) :
    _bias(0.0)
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
    if (!st)
        return;
    _SVCoeff = HomogenNumericTable<modelFPType>::create(NULL, 1, 0, &st);
    if (!st)
        return;
    _SVIndices = HomogenNumericTable<int>::create(NULL, 1, 0, &st);
    if (!st)
        return;
}

template DAAL_EXPORT services::SharedPtr<Model> Model::create<DAAL_FPTYPE>(size_t nColumns,
                                         data_management::NumericTableIface::StorageLayout layout,
                                         services::Status *stat);

template DAAL_EXPORT Model::Model(DAAL_FPTYPE, size_t, data_management::NumericTableIface::StorageLayout, services::Status&);

} // namespace interface1=
} // namespace svm
} // namespace algorithms
} // namespace daal
