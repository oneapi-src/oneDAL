/* file: decision_tree_regression_predict_batch.h */
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
//  Implementation of the class defining the decision tree model
//--
*/

#ifndef __DECISION_TREE_REGRESSION_PREDICT_BATCH_
#define __DECISION_TREE_REGRESSION_PREDICT_BATCH_

#include "algorithms/decision_tree/decision_tree_regression_predict_types.h"

namespace daal
{
namespace algorithms
{
namespace decision_tree
{
namespace regression
{
namespace prediction
{
namespace interface1
{

template <typename algorithmFPType>
DAAL_EXPORT services::Status Result::allocate(const daal::algorithms::Input * input, const daal::algorithms::Parameter * par, const int method)
{
    const size_t nVectors = (static_cast<const Input *>(input))->get(data)->getNumberOfRows();

    services::Status st;
    set(prediction, data_management::HomogenNumericTable<algorithmFPType>::create(1, nVectors, data_management::NumericTableIface::doAllocate, &st));
    return st;
}

} // namespace interface1
} // namespace prediction
} // namespace regression
} // namespace ridge_regression
} // namespace algorithms
} // namespace daal

#endif
