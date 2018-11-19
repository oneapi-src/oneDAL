/* file: naivebayes_train_partial_result.h */
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
 * Allocates memory for storing partial results of the naive Bayes training algorithm
 * \param[in] input        Pointer to input object
 * \param[in] parameter    Pointer to parameter
 * \param[in] method       Computation method
 */
template <typename algorithmFPType>
DAAL_EXPORT Status PartialResult::initialize(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method)
{
    get(classifier::training::partialModel)->initialize<algorithmFPType>();
    return Status();
}

/**
 * Allocates memory for storing partial results of the naive Bayes training algorithm
 * \param[in] input        Pointer to input object
 * \param[in] parameter    Pointer to parameter
 * \param[in] method       Computation method
 */
template <typename algorithmFPType>
DAAL_EXPORT Status PartialResult::allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method)
{
    const classifier::training::InputIface *algInput = static_cast<const classifier::training::InputIface *>(input);
    const Parameter *algPar = static_cast<const Parameter *>(parameter);
    size_t nFeatures = algInput->getNumberOfFeatures();
    Status st;
    PartialModelPtr partialModelPtr = PartialModel::create<algorithmFPType>(nFeatures, *algPar, &st);
    DAAL_CHECK_STATUS_VAR(st);
    set(classifier::training::partialModel, partialModelPtr);
    return st;
}

}// namespace interface1
}// namespace training
}// namespace multinomial_naive_bayes
}// namespace algorithms
}// namespace daal
