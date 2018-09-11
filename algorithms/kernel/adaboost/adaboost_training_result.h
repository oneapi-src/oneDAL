/* file: adaboost_training_result.h */
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
//  Implementation of Ada Boost training algorithm interface.
//--
*/

#ifndef __ADABOOST_TRAINING_RESULT_
#define __ADABOOST_TRAINING_RESULT_

#include "algorithms/boosting/adaboost_training_types.h"

namespace daal
{
namespace algorithms
{
namespace adaboost
{
namespace training
{

/**
 * Allocates memory to store final results of AdaBoost training
 * \param[in] input         %Input of the AdaBoost training algorithm
 * \param[in] parameter     Parameters of the algorithm
 * \param[in] method        AdaBoost computation method
 */
template <typename algorithmFPType>
DAAL_EXPORT services::Status Result::allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method)
{
    services::Status s;
    const classifier::training::Input *algInput = static_cast<const classifier::training::Input *>(input);
    set(classifier::training::model, Model::create<algorithmFPType>(algInput->getNumberOfFeatures(), &s));
    return s;
}

} // namespace training
} // namespace adaboost
} // namespace algorithms
} // namespace daal

#endif
