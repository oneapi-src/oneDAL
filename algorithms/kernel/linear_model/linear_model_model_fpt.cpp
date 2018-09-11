/* file: linear_model_model_fpt.cpp */
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

#include "linear_model_model_impl.h"
#include "data_management/data/homogen_numeric_table.h"

namespace daal
{
namespace algorithms
{
namespace linear_model
{
namespace internal
{
using namespace daal::data_management;

template<typename modelFPType>
ModelInternal::ModelInternal(size_t nFeatures, size_t nResponses, const Parameter &par, modelFPType dummy) :
    _interceptFlag(par.interceptFlag)
{
    services::Status st;
    _beta = HomogenNumericTable<modelFPType>::create(nFeatures + 1, nResponses, NumericTable::doAllocate, 0, &st);
    if (!st) return;
}

template ModelInternal::ModelInternal(size_t nFeatures, size_t nResponses, const Parameter &par, DAAL_FPTYPE dummy);
}
}
}
}
