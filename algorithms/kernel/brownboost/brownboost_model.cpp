/* file: brownboost_model.cpp */
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
//  Implementation of class defining Brown Boost model.
//--
*/

#include "algorithms/boosting/brownboost_model.h"
#include "serialization_utils.h"
#include "daal_strings.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace brownboost
{
namespace interface1
{
__DAAL_REGISTER_SERIALIZATION_CLASS(Model, SERIALIZATION_BROWNBOOST_MODEL_ID);

/** Default constructor */
Parameter::Parameter() : boosting::Parameter(), accuracyThreshold(0.3), maxIterations(10),
    newtonRaphsonAccuracyThreshold(1.0e-3), newtonRaphsonMaxIterations(100), degenerateCasesThreshold(1.0e-2) {}

/**
 * Constructs BrownBoost parameter structure
 * \param[in] wlTrain       Pointer to the training algorithm of the weak learner
 * \param[in] wlPredict     Pointer to the prediction algorithm of the weak learner
 * \param[in] acc           Accuracy of the BrownBoost training algorithm
 * \param[in] maxIter       Maximal number of iterations of the BrownBoost training algorithm
 * \param[in] nrAcc         Accuracy threshold for Newton-Raphson iterations in the BrownBoost training algorithm
 * \param[in] nrMaxIter     Maximal number of Newton-Raphson iterations in the BrownBoost training algorithm
 * \param[in] dcThreshold          Threshold needed  to avoid degenerate cases in the BrownBoost training algorithm
 */
Parameter::Parameter(services::SharedPtr<weak_learner::training::Batch>   wlTrain,
          services::SharedPtr<weak_learner::prediction::Batch> wlPredict,
          double acc, size_t maxIter, double nrAcc, size_t nrMaxIter, double dcThreshold) :
    boosting::Parameter(wlTrain, wlPredict), accuracyThreshold(acc), maxIterations(maxIter),
    newtonRaphsonAccuracyThreshold(nrAcc), newtonRaphsonMaxIterations(nrMaxIter), degenerateCasesThreshold(dcThreshold) {}

services::Status Parameter::check() const
{
    services::Status s = boosting::Parameter::check();
    if(!s) return s;
    DAAL_CHECK_EX(accuracyThreshold > 0 && accuracyThreshold < 1, ErrorIncorrectParameter, ParameterName, accuracyThresholdStr());
    DAAL_CHECK_EX(maxIterations > 0, ErrorIncorrectParameter, ParameterName, maxIterationsStr());
    DAAL_CHECK_EX(newtonRaphsonAccuracyThreshold > 0 && newtonRaphsonAccuracyThreshold < 1, ErrorIncorrectParameter, ParameterName, newtonRaphsonAccuracyThresholdStr());
    DAAL_CHECK_EX(newtonRaphsonMaxIterations > 0, ErrorIncorrectParameter, ParameterName, newtonRaphsonMaxIterationsStr());
    DAAL_CHECK_EX(degenerateCasesThreshold > 0 && degenerateCasesThreshold < 1, ErrorIncorrectParameter, ParameterName, degenerateCasesThresholdStr());
    return s;
}

/**
 *  Returns a pointer to the array of weights of weak learners constructed
 *  during training of the brownBoost algorithm.
 *  The size of the array equals the number of weak learners
 *  \return Array of weights of weak learners.
 */
NumericTablePtr Model::getAlpha()
{
    return _alpha;
}


} // namespace interface1
} // namespace brownboost
} // namespace algorithms
} // namespace daal
