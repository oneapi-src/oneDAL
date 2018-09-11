/* file: logistic_fpt.cpp */
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
//  Implementation of logistic algorithm and types methods.
//--
*/

#include "logistic_types.h"

using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace math
{
namespace logistic
{
namespace interface1
{
/**
 * Allocates memory to store the result of the logistic function
 * \param[in] input  %Input object for the logistic function
 * \param[in] par    %Parameter of the logistic function
 * \param[in] method Computation method of the logistic function
 */
template <typename algorithmFPType>
DAAL_EXPORT Status Result::allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *par, const int method)
{
    Input *algInput = static_cast<Input *>(const_cast<daal::algorithms::Input *>(input));

    DAAL_CHECK(algInput, ErrorNullInput);
    DAAL_CHECK(algInput->get(data).get(), ErrorNullInputNumericTable);

    const size_t nFeatures     = algInput->get(data)->getNumberOfColumns();
    const size_t nObservations = algInput->get(data)->getNumberOfRows();
    Status st;
    Argument::set(value, data_management::HomogenNumericTable<algorithmFPType>::create(nFeatures, nObservations, data_management::NumericTable::doAllocate, &st));
    return st;
}

template DAAL_EXPORT Status Result::allocate<DAAL_FPTYPE>(const daal::algorithms::Input *input, const daal::algorithms::Parameter *par, const int method);

}// namespace interface1
}// namespace logistic
}// namespace math
}// namespace algorithms
}// namespace daal
