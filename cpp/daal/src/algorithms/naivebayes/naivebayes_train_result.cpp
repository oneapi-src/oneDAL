/* file: naivebayes_train_result.cpp */
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
#include "src/services/serialization_utils.h"
#include "src/services/daal_strings.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace multinomial_naive_bayes
{
__DAAL_REGISTER_SERIALIZATION_CLASS(Model, SERIALIZATION_NAIVE_BAYES_MODEL_ID);

namespace training
{
__DAAL_REGISTER_SERIALIZATION_CLASS(Result, SERIALIZATION_NAIVE_BAYES_RESULT_ID);
Result::Result() {}

/**
 * Returns the model trained with the naive Bayes training algorithm
 * \param[in] id    Identifier of the result, \ref classifier::training::ResultId
 * \return          Model trained with the classification algorithm
 */
multinomial_naive_bayes::ModelPtr Result::get(classifier::training::ResultId id) const
{
    return services::staticPointerCast<multinomial_naive_bayes::Model, data_management::SerializationIface>(Argument::get(id));
}

/**
* Checks the correctness of Result object
* \param[in] partialResult Pointer to the partial results structure
* \param[in] parameter     Parameter of the algorithm
* \param[in] method        Computation method
*/
services::Status Result::check(const daal::algorithms::PartialResult * partialResult, const daal::algorithms::Parameter * parameter, int method) const
{
    Status s;

    const PartialResult * pres = static_cast<const PartialResult *>(partialResult);
    size_t nFeatures           = pres->getNumberOfFeatures();

    DAAL_CHECK_STATUS(s, checkImpl(nFeatures, parameter));

    return s;
}

/**
 * Checks the final result of the naive Bayes training algorithm
 * \param[in] input      %Input of algorithm
 * \param[in] parameter  %Parameter of algorithm
 * \param[in] method     Computation method
 */
services::Status Result::check(const daal::algorithms::Input * input, const daal::algorithms::Parameter * parameter, int method) const
{
    Status s;
    DAAL_CHECK_STATUS(s, classifier::training::Result::checkImpl(input, parameter));

    const classifier::training::InputIface * algInput = static_cast<const classifier::training::InputIface *>(input);

    size_t nFeatures = algInput->getNumberOfFeatures();

    DAAL_CHECK_STATUS(s, checkImpl(nFeatures, parameter));

    return s;
}

services::Status Result::checkImpl(size_t nFeatures, const daal::algorithms::Parameter * parameter) const
{
    Status s;
    ModelPtr resModel = get(classifier::training::model);
    DAAL_CHECK(resModel, ErrorNullModel);

    const size_t trainingDataFeatures = resModel->getNFeatures();
    DAAL_CHECK(trainingDataFeatures, services::ErrorModelNotFullInitialized);

    size_t nClasses = 0;
    NumericTablePtr alphaTable;
    const multinomial_naive_bayes::Parameter * algPar = dynamic_cast<const multinomial_naive_bayes::Parameter *>(parameter);
    if (algPar)
    {
        nClasses   = algPar->nClasses;
        alphaTable = algPar->alpha;
    }

    s |= checkNumericTable(resModel->getLogP().get(), logPStr(), 0, 0, 1, nClasses);
    s |= checkNumericTable(resModel->getLogTheta().get(), logThetaStr(), 0, 0, nFeatures, nClasses);

    if (alphaTable)
    {
        s |= checkNumericTable(alphaTable.get(), alphaStr(), 0, 0, nFeatures, 1);
    }

    return s;
}

} // namespace training
} // namespace multinomial_naive_bayes
} // namespace algorithms
} // namespace daal
