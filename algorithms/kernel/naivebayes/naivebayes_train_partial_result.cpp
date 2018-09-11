/* file: naivebayes_train_partial_result.cpp */
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
__DAAL_REGISTER_SERIALIZATION_CLASS(PartialModel, SERIALIZATION_NAIVE_BAYES_PARTIALMODEL_ID);
}

namespace training
{
namespace interface1
{
__DAAL_REGISTER_SERIALIZATION_CLASS(PartialResult, SERIALIZATION_NAIVE_BAYES_PARTIAL_RESULT_ID);

PartialResult::PartialResult() {}

/**
 * Returns the partial model trained with the classification algorithm
 * \param[in] id    Identifier of the partial model, \ref classifier::training::PartialResultId
 * \return          Model trained with the classification algorithm
 */
multinomial_naive_bayes::PartialModelPtr PartialResult::get(classifier::training::PartialResultId id) const
{
    return services::staticPointerCast<multinomial_naive_bayes::PartialModel, data_management::SerializationIface>(Argument::get(id));
}

/**
* Returns number of columns in the naive Bayes partial result
* \return Number of columns in the partial result
*/
size_t PartialResult::getNumberOfFeatures() const
{
    PartialModelPtr ntPtr =
        services::staticPointerCast<PartialModel, data_management::SerializationIface>(Argument::get(classifier::training::partialModel));
    return ntPtr ? ntPtr->getNFeatures() : 0;
}

/**
 * Checks partial result of the naive Bayes training algorithm
 * \param[in] input      Algorithm %input object
 * \param[in] parameter  Algorithm %parameter
 * \param[in] method     Computation method
 */
services::Status PartialResult::check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, int method) const
{
    services::Status s;
    DAAL_CHECK_STATUS(s, classifier::training::PartialResult::checkImpl(input, parameter));

    const classifier::training::InputIface *algInput = static_cast<const classifier::training::InputIface *>(input);

    size_t nFeatures = algInput->getNumberOfFeatures();
    DAAL_CHECK_STATUS(s, checkImpl(nFeatures,parameter));

    return s;
}

/**
* Checks partial result of the naive Bayes training algorithm
* \param[in] parameter  Algorithm %parameter
* \param[in] method     Computation method
*/
services::Status PartialResult::check(const daal::algorithms::Parameter *parameter, int method)  const
{
    services::Status s;
    size_t nFeatures = getNumberOfFeatures();

    DAAL_CHECK_STATUS(s, checkImpl(nFeatures,parameter));

    return s;
}

services::Status PartialResult::checkImpl(size_t nFeatures,const daal::algorithms::Parameter* parameter) const
{
    services::Status s;
    PartialModelPtr resModel = get(classifier::training::partialModel);
    DAAL_CHECK(resModel, ErrorNullModel);
    const size_t trainingDataFeatures = resModel->getNFeatures();
    DAAL_CHECK(trainingDataFeatures, services::ErrorModelNotFullInitialized);
    const multinomial_naive_bayes::Parameter *algPar = static_cast<const multinomial_naive_bayes::Parameter *>(parameter);
    size_t nClasses = algPar->nClasses;

    s |= checkNumericTable(resModel->getClassSize().get(), classSizeStr(), 0, 0, 1, nClasses);
    s |= checkNumericTable(resModel->getClassGroupSum().get(), groupSumStr(), 0, 0, nFeatures, nClasses);

    return s;
}


}// namespace interface1
}// namespace training
}// namespace multinomial_naive_bayes
}// namespace algorithms
}// namespace daal
