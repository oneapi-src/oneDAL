/* file: df_regression_predict_result.h */
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
//  Implementation of the decision forest algorithm interface
//--
*/

#ifndef __DF_REGRESSION_PREDICT_RESULT_H_
#define __DF_REGRESSION_PREDICT_RESULT_H_

#include "algorithms/decision_forest/decision_forest_regression_predict_types.h"

namespace daal
{
namespace algorithms
{
namespace decision_forest
{
namespace regression
{
namespace prediction
{

/**
 * Allocates memory to store a partial result of decision forest model-based prediction
 * \param[in] input   %Input object
 * \param[in] par     %Parameter of the algorithm
 * \param[in] method  Algorithm method
 */
template <typename algorithmFPType>
DAAL_EXPORT services::Status Result::allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *par, const int method)
{
    size_t nVectors = (static_cast<const Input *>(input))->get(data)->getNumberOfRows();
    services::Status st;
    set(prediction, data_management::HomogenNumericTable<algorithmFPType>::create(1, nVectors, data_management::NumericTableIface::doAllocate, &st));
    return st;
}

} // namespace prediction
} // namespace regression
} // namespace decision_forest
} // namespace algorithms
} // namespace daal

#endif
