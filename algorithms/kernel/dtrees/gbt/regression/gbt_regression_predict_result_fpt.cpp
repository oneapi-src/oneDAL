/* file: gbt_regression_predict_result_fpt.cpp */
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
//  Implementation of the gradient boosted trees regression algorithm interface
//--
*/

#include "algorithms/gradient_boosted_trees/gbt_regression_predict_types.h"
#include "data_management/data/homogen_numeric_table.h"
#include "daal_strings.h"

namespace daal
{
namespace algorithms
{
namespace gbt
{
namespace regression
{
namespace prediction
{

using namespace daal::services;

template <typename algorithmFPType>
DAAL_EXPORT services::Status Result::allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *par, const int method)
{
    const Input* algInput= (static_cast<const Input *>(input));
    data_management::NumericTablePtr dataPtr = algInput->get(data);
    DAAL_CHECK_EX(dataPtr.get(), ErrorNullInputNumericTable, ArgumentName, dataStr());
    services::Status s;
    const size_t nVectors = dataPtr->getNumberOfRows();
    Argument::set(prediction,
        data_management::HomogenNumericTable<algorithmFPType>::create(1, nVectors, data_management::NumericTableIface::doAllocate, &s));
    return s;
}

template DAAL_EXPORT services::Status Result::allocate<DAAL_FPTYPE>(const daal::algorithms::Input *input, const daal::algorithms::Parameter *par, const int method);

}// namespace prediction
}// namespace regression
}// namespace gbt
}// namespace algorithms
}// namespace daal
