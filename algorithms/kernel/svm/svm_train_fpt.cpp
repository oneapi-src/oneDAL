/* file: svm_train_fpt.cpp */
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
//  Implementation of cholesky algorithm and types methods.
//--
*/

#include "svm_train_types.h"

namespace daal
{
namespace algorithms
{
namespace svm
{
namespace training
{
namespace interface1
{
/**
 * Allocates memory for storing SVM training results
 * \param[in] input     Pointer to input structure
 * \param[in] parameter Pointer to parameter structure
 * \param[in] method    Algorithm method
 */
template <typename algorithmFPType>
DAAL_EXPORT services::Status Result::allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method)
{
    const classifier::training::Input *algInput = static_cast<const classifier::training::Input *>(input);

    algorithmFPType dummy = 1.0;
    services::Status st;
    set(classifier::training::model, svm::Model::create<algorithmFPType>(algInput->get(classifier::training::data)->getNumberOfColumns(), algInput->get(classifier::training::data)->getDataLayout(), &st));
    return st;
}

template DAAL_EXPORT services::Status Result::allocate<DAAL_FPTYPE>(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method);

}// namespace interface1
}// namespace svm
}// namespace cholesky
}// namespace algorithms
}// namespace daal
