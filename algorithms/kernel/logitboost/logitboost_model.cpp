/* file: logitboost_model.cpp */
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
//  Implementation of class defining LogitBoost model.
//--
*/

#include "algorithms/boosting/logitboost_model.h"
#include "serialization_utils.h"
#include "daal_strings.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace logitboost
{
namespace interface1
{

__DAAL_REGISTER_SERIALIZATION_CLASS(Model, SERIALIZATION_LOGITBOOST_MODEL_ID);
/** Default constructor */
Parameter::Parameter() : boosting::Parameter(), accuracyThreshold(0.0), maxIterations(10), nClasses(0),
    weightsDegenerateCasesThreshold(1e-10), responsesDegenerateCasesThreshold(1e-10) {}

/**
 * Constructs LogitBoost parameter structure
 * \param[in] wlTrain       Pointer to the training algorithm of the weak learner
 * \param[in] wlPredict     Pointer to the prediction algorithm of the weak learner
 * \param[in] acc           Accuracy of the LogitBoost training algorithm
 * \param[in] maxIter       Maximal number of terms in additive regression
 * \param[in] nC            Number of classes in the training data set
 * \param[in] wThr          Threshold to avoid degenerate cases when calculating weights W
 * \param[in] zThr          Threshold to avoid degenerate cases when calculating responses Z
 */
Parameter::Parameter(const SharedPtr<weak_learner::training::Batch>&   wlTrain,
    const SharedPtr<weak_learner::prediction::Batch>& wlPredict,
          double acc, size_t maxIter, size_t nC, double wThr, double zThr) :
    boosting::Parameter(wlTrain, wlPredict),
    accuracyThreshold(acc), maxIterations(maxIter), nClasses(nC), weightsDegenerateCasesThreshold(wThr), responsesDegenerateCasesThreshold(zThr) {}

services::Status Parameter::check() const
{
    services::Status s = boosting::Parameter::check();
    if(!s) return s;
    DAAL_CHECK_EX(accuracyThreshold >= 0 && accuracyThreshold < 1, ErrorIncorrectParameter, ParameterName, accuracyThresholdStr());
    DAAL_CHECK_EX(maxIterations > 0, ErrorIncorrectParameter, ParameterName, maxIterationsStr());
    DAAL_CHECK_EX(nClasses >= 2, ErrorIncorrectParameter, ParameterName, nClassesStr());
    DAAL_CHECK_EX(weightsDegenerateCasesThreshold > 0, ErrorIncorrectParameter, ParameterName, weightsDegenerateCasesThresholdStr());
    DAAL_CHECK_EX(responsesDegenerateCasesThreshold > 0, ErrorIncorrectParameter, ParameterName, responsesDegenerateCasesThresholdStr());
    return s;
}


Model::Model(size_t nFeatures, const Parameter *par, services::Status &st) :
    boosting::Model(nFeatures, st),
    _nIterations(par->maxIterations) { }

/**
 * Constructs the LogitBoost model
 * \param[in]  nFeatures Number of features in the dataset
 * \param[in]  par       Pointer to the parameter structure of the LogitBoost algorithm
 * \param[out] stat      Status of the model construction
 */
ModelPtr Model::create(size_t nFeatures, const Parameter *par, services::Status *stat)
{
    DAAL_DEFAULT_CREATE_IMPL_EX(Model, nFeatures, par);
}

/**
 * Sets the number of iterations for the algorithm
 * @param nIterations   Number of iterations
 */
void Model::setIterations(size_t nIterations)
{
    _nIterations = nIterations;
}

/**
 * Returns the number of iterations done by the training algorithm
 * \return The number of iterations done by the training algorithm
 */
size_t Model::getIterations() const
{
    return _nIterations;
}


} // namespace interface1
} // namespace logitboost
} // namespace algorithms
} // namespace daal
