/* file: multiclassclassifier_train_fpt.cpp */
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
//  Implementation of multi class classifier algorithm and types methods.
//--
*/

#include "multi_class_classifier_train_types.h"

namespace daal
{
namespace algorithms
{
namespace multi_class_classifier
{
namespace training
{
namespace interface1
{
/**
 * Registers user-allocated memory to store the results of the multi-class classifier training decomposition
 * \param[in] input       Pointer to the structure with input objects
 * \param[in] parameter   Pointer to the structure with algorithm parameters
 * \param[in] method      Computation method
 */
template <typename algorithmFPType>
DAAL_EXPORT services::Status Result::allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method)
{
    const ParameterBase *algParameter = static_cast<const ParameterBase *>(parameter);
    const classifier::training::Input *algInput = static_cast<const classifier::training::Input *>(input);
    services::Status st;
    ModelPtr modelPtr = Model::create(algInput->getNumberOfFeatures(), algParameter, &st);
    DAAL_CHECK_STATUS_VAR(st);
    set(classifier::training::model, modelPtr);
    return st;
}

template DAAL_EXPORT services::Status Result::allocate<DAAL_FPTYPE>(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method);

}// namespace interface1
}// namespace training
}// namespace multi_class_classifier
}// namespace algorithms
}// namespace daal
