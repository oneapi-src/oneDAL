/* file: naivebayes_train_result_fpt.cpp */
/*******************************************************************************
* Copyright 2014 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

/*
//++
//  Implementation of multinomial naive bayes algorithm and types methods.
//--
*/

#include "algorithms/naive_bayes/multinomial_naive_bayes_training_types.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace multinomial_naive_bayes
{
namespace training
{
/**
 * Allocates memory for storing final result computed with naive Bayes training algorithm
 * \param[in] input      Pointer to input object
 * \param[in] parameter  Pointer to parameter
 * \param[in] method     Computation method
 */
template <typename algorithmFPType>
DAAL_EXPORT services::Status Result::allocate(const daal::algorithms::Input * input, const daal::algorithms::Parameter * parameter, const int method)
{
    const classifier::training::InputIface * algInput = static_cast<const classifier::training::InputIface *>(input);
    size_t nFeatures                                  = algInput->getNumberOfFeatures();
    services::Status st;
    ModelPtr modelPtr;

    const multinomial_naive_bayes::Parameter * algPar = dynamic_cast<const multinomial_naive_bayes::Parameter *>(parameter);
    if (algPar) modelPtr = Model::create<algorithmFPType>(nFeatures, *algPar, &st);
    DAAL_CHECK(modelPtr, ErrorNullModel);
    DAAL_CHECK_STATUS_VAR(st);
    set(classifier::training::model, modelPtr);
    return st;
}

/**
* Allocates memory for storing final result computed with naive Bayes training algorithm
* \param[in] partialResult      Pointer to partial result structure
* \param[in] parameter          Pointer to parameter structure
* \param[in] method             Computation method
*/
template <typename algorithmFPType>
DAAL_EXPORT services::Status Result::allocate(const daal::algorithms::PartialResult * partialResult, const daal::algorithms::Parameter * parameter,
                                              const int method)
{
    const PartialResult * pres = static_cast<const PartialResult *>(partialResult);
    size_t nFeatures           = pres->getNumberOfFeatures();
    services::Status st;
    ModelPtr modelPtr;

    const multinomial_naive_bayes::Parameter * algPar = dynamic_cast<const multinomial_naive_bayes::Parameter *>(parameter);
    if (algPar) modelPtr = Model::create<algorithmFPType>(nFeatures, *algPar, &st);

    DAAL_CHECK(modelPtr, ErrorNullModel);
    DAAL_CHECK_STATUS_VAR(st);
    set(classifier::training::model, modelPtr);
    return st;
}

template DAAL_EXPORT services::Status Result::allocate<DAAL_FPTYPE>(const daal::algorithms::Input * input,
                                                                    const daal::algorithms::Parameter * parameter, const int method);
template DAAL_EXPORT services::Status Result::allocate<DAAL_FPTYPE>(const daal::algorithms::PartialResult * partialResult,
                                                                    const daal::algorithms::Parameter * parameter, const int method);

} // namespace training
} // namespace multinomial_naive_bayes
} // namespace algorithms
} // namespace daal
