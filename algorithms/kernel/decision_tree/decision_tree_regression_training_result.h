/* file: decision_tree_regression_training_result.h */
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
//  Implementation of the class defining the Decision tree model
//--
*/

#ifndef __DECISION_TREE_REGRESSION_TRAINING_RESULT_
#define __DECISION_TREE_REGRESSION_TRAINING_RESULT_

#include "algorithms/decision_tree/decision_tree_regression_training_types.h"

namespace daal
{
namespace algorithms
{
namespace decision_tree
{
namespace regression
{
namespace training
{

/**
 * Allocates memory to store the result of Decision tree model-based training
 * \param[in] input Pointer to an object containing the input data
 * \param[in] parameter %Parameter of Decision tree model-based training
 * \param[in] method Computation method for the algorithm
 */
template<typename algorithmFPType>
DAAL_EXPORT services::Status Result::allocate(const daal::algorithms::Input * input, const Parameter * parameter, int method)
{
    services::Status status;
    set(algorithms::regression::training::model, Model::create(&status));
    return status;
}

} // namespace training
} // namespace regression
} // namespace decision_tree
} // namespace algorithms
} // namespace daal

#endif
