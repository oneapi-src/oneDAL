/* file: classifier_predict_fpt.cpp */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation.
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
//  Implementation of classifier prediction Result.
//--
*/

#include "classifier_predict_types.h"

namespace daal
{
namespace algorithms
{
namespace classifier
{
namespace prediction
{
namespace interface1
{

/**
 * Allocates memory for storing prediction results of the classification algorithm
 * \tparam  algorithmFPType     Data type for storing prediction results
 * \param[in] input     Pointer to the input objects of the classification algorithm
 * \param[in] parameter Pointer to the parameters of the classification algorithm
 * \param[in] method    Computation method
 */
template <typename algorithmFPType>
DAAL_EXPORT services::Status Result::allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method)
{
    services::Status st;
    set(prediction, data_management::HomogenNumericTable<algorithmFPType>::create(1, (static_cast<const InputIface *>(input))->getNumberOfRows(), data_management::NumericTableIface::doAllocate, &st));
    return st;
}

template DAAL_EXPORT services::Status Result::allocate<DAAL_FPTYPE>(const daal::algorithms::Input *input, const daal::algorithms::Parameter *par, const int method);

}
}
}
}
}
