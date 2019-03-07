/* file: stump_train_fpt.cpp */
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
//  Implementation of stump algorithm and types methods.
//--
*/

#include "stump_training_types.h"

namespace daal
{
namespace algorithms
{
namespace stump
{
namespace training
{
namespace interface1
{
/**
 * Allocates memory to store final results of the decision stump training algorithm
 * \tparam algorithmFPType  Data type to store prediction results
 * \param[in] input         %Input objects for the decision stump training algorithm
 * \param[in] parameter     Parameters of the decision stump training algorithm
 * \param[in] method        Decision stump training method
 */
template <typename algorithmFPType>
DAAL_EXPORT services::Status Result::allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method)
{
    algorithmFPType dummy = 1.0;
    const classifier::training::InputIface *algInput = static_cast<const classifier::training::InputIface *>(input);
    services::Status st;
    stump::ModelPtr modelPtr = stump::Model::create<algorithmFPType>(algInput->getNumberOfFeatures(), &st);
    DAAL_CHECK_STATUS_VAR(st);
    classifier::training::Result::set(classifier::training::model, modelPtr);
    return st;
}

template DAAL_EXPORT services::Status Result::allocate<DAAL_FPTYPE>(const daal::algorithms::Input *input, const daal::algorithms::Parameter *par, const int method);

}// namespace interface1
}// namespace training
}// namespace stump
}// namespace algorithms
}// namespace daal
