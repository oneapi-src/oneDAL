/* file: naivebayes_train_result_fpt.cpp */
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
namespace interface1
{
/**
 * Allocates memory for storing final result computed with naive Bayes training algorithm
 * \param[in] input      Pointer to input object
 * \param[in] parameter  Pointer to parameter
 * \param[in] method     Computation method
 */
template <typename algorithmFPType>
DAAL_EXPORT services::Status Result::allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method)
{
    const classifier::training::InputIface *algInput = static_cast<const classifier::training::InputIface *>(input);
    Parameter *algPar = static_cast<Parameter *>(const_cast<daal::algorithms::Parameter *>(parameter));
    size_t nFeatures = algInput->getNumberOfFeatures();
    services::Status st;
    ModelPtr modelPtr = Model::create<algorithmFPType>(nFeatures, *algPar, &st);
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
DAAL_EXPORT services::Status Result::allocate(const daal::algorithms::PartialResult *partialResult, const daal::algorithms::Parameter *parameter, const int method)
{
    const PartialResult *pres = static_cast<const PartialResult *>(partialResult);
    Parameter *algPar = static_cast<Parameter *>(const_cast<daal::algorithms::Parameter *>(parameter));
    size_t nFeatures = pres->getNumberOfFeatures();
    services::Status st;
    ModelPtr modelPtr = Model::create<algorithmFPType>(nFeatures, *algPar, &st);
    DAAL_CHECK_STATUS_VAR(st);
    set(classifier::training::model, modelPtr);
    return st;
}

template DAAL_EXPORT services::Status Result::allocate<DAAL_FPTYPE>(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method);
template DAAL_EXPORT services::Status Result::allocate<DAAL_FPTYPE>(const daal::algorithms::PartialResult *partialResult, const daal::algorithms::Parameter *parameter, const int method);

}// namespace interface1
}// namespace training
}// namespace multinomial_naive_bayes
}// namespace algorithms
}// namespace daal
