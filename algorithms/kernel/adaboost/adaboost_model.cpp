/* file: adaboost_model.cpp */
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
//  Implementation of class defining Ada Boost model
//--
*/

#include "algorithms/boosting/adaboost_model.h"
#include "serialization_utils.h"
#include "daal_strings.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace adaboost
{
namespace interface1
{
__DAAL_REGISTER_SERIALIZATION_CLASS(Model, SERIALIZATION_ADABOOST_MODEL_ID);

Parameter::Parameter() : boosting::Parameter(), accuracyThreshold(0.0), maxIterations(10) {}

/**
 * Constructs the AdaBoost parameter structure
 * \param[in] wlTrain       Pointer to the training algorithm of the weak learner
 * \param[in] wlPredict     Pointer to the prediction algorithm of the weak learner
 * \param[in] acc           Accuracy of the AdaBoost training algorithm
 * \param[in] maxIter       Maximal number of iterations of the AdaBoost training algorithm
 */
Parameter::Parameter(SharedPtr<weak_learner::training::Batch>   wlTrain,
          SharedPtr<weak_learner::prediction::Batch> wlPredict,
          double acc, size_t maxIter) :
    boosting::Parameter(wlTrain, wlPredict),
    accuracyThreshold(acc), maxIterations(maxIter) {}

Status Parameter::check() const
{
    Status s;
    DAAL_CHECK_STATUS(s, boosting::Parameter::check());
    DAAL_CHECK_EX(accuracyThreshold >= 0 && accuracyThreshold < 1, ErrorIncorrectParameter, ParameterName, accuracyThresholdStr());
    DAAL_CHECK_EX(maxIterations > 0, ErrorIncorrectParameter, ParameterName, maxIterationsStr());
    return s;
}

/**
 *  Returns a pointer to the array of weights of weak learners constructed
 *  during training of the AdaBoost algorithm.
 *  The size of the array equals the number of weak learners
 *  \return Array of weights of weak learners.
 */
NumericTablePtr Model::getAlpha() const
{
    return _alpha;
}


} // namespace interface1
} // namespace adaboost
} // namespace algorithms
} // namespace daal
