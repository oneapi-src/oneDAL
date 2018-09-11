/* file: naivebayes_train_result.cpp */
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
//  Implementation of multinomial naive bayes algorithm and types methods.
//--
*/

#include "multinomial_naive_bayes_training_types.h"
#include "serialization_utils.h"
#include "daal_strings.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace multinomial_naive_bayes
{

namespace interface1
{
__DAAL_REGISTER_SERIALIZATION_CLASS(Model, SERIALIZATION_NAIVE_BAYES_MODEL_ID);
}

namespace training
{
namespace interface1
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
services::Status Result::check(const daal::algorithms::PartialResult *partialResult, const daal::algorithms::Parameter *parameter, int method) const
{
    Status s;

    const PartialResult *pres = static_cast<const PartialResult *>(partialResult);
    size_t nFeatures = pres->getNumberOfFeatures();

    DAAL_CHECK_STATUS(s,checkImpl(nFeatures,parameter));

    return s;
}

/**
 * Checks the final result of the naive Bayes training algorithm
 * \param[in] input      %Input of algorithm
 * \param[in] parameter  %Parameter of algorithm
 * \param[in] method     Computation method
 */
services::Status Result::check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, int method) const
{
    Status s;
    DAAL_CHECK_STATUS(s, classifier::training::Result::checkImpl(input, parameter));

    const classifier::training::InputIface *algInput = static_cast<const classifier::training::InputIface *>(input);

    size_t nFeatures = algInput->getNumberOfFeatures();

    DAAL_CHECK_STATUS(s, checkImpl(nFeatures,parameter));

    return s;
}

services::Status Result::checkImpl(size_t nFeatures,const daal::algorithms::Parameter* parameter) const
{
    Status s;
    ModelPtr resModel = get(classifier::training::model);
    DAAL_CHECK(resModel, ErrorNullModel);

    const size_t trainingDataFeatures = resModel->getNFeatures();
    DAAL_CHECK(trainingDataFeatures, services::ErrorModelNotFullInitialized);

    const multinomial_naive_bayes::Parameter *algPar = static_cast<const multinomial_naive_bayes::Parameter *>(parameter);
    size_t nClasses = algPar->nClasses;

    s |= checkNumericTable(resModel->getLogP().get(), logPStr(), 0, 0, 1, nClasses);
    s |= checkNumericTable(resModel->getLogTheta().get(), logThetaStr(), 0, 0, nFeatures, nClasses);

    if(algPar->alpha)
    {
        s |= checkNumericTable(algPar->alpha.get(), alphaStr(), 0, 0, nFeatures, 1);
    }

    return s;
}

}// namespace interface1
}// namespace training
}// namespace multinomial_naive_bayes
}// namespace algorithms
}// namespace daal
