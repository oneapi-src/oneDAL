/* file: ridge_regression_training_result.h */
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
//  Implementation of the class defining the ridge regression model
//--
*/

#ifndef __RIDGE_REGRESSION_TRAINING_RESULT_H__
#define __RIDGE_REGRESSION_TRAINING_RESULT_H__

#include "algorithms/ridge_regression/ridge_regression_training_types.h"
#include "ridge_regression_ne_model_impl.h"

namespace daal
{
namespace algorithms
{
namespace ridge_regression
{
namespace training
{
using namespace daal::services;

/**
 * Allocates memory to store the result of ridge regression model-based training
 * \param[in] input Pointer to an object containing the input data
 * \param[in] parameter %Parameter of ridge regression model-based training
 * \param[in] method Computation method for the algorithm
 */
template<typename algorithmFPType>
DAAL_EXPORT Status Result::allocate(const daal::algorithms::Input * input, const Parameter * parameter, const int method)
{
    const Input * const in = static_cast<const Input *>(input);

    Status s;
    if (method == normEqDense)
    {
        const algorithmFPType dummy = 1.0;
        set(model, ridge_regression::ModelPtr(new ridge_regression::internal::ModelNormEqImpl(in->getNumberOfFeatures(), in->getNumberOfDependentVariables(),
                                                                *parameter, dummy, s)));
    }

    return s;
}

// *
//  * Allocates memory to store the result of ridge regression model-based training
//  * \param[in] partialResult Pointer to an object containing the input data
//  * \param[in] method        Computation method of the algorithm
//  * \param[in] parameter     %Parameter of ridge regression model-based training

template<typename algorithmFPType>
DAAL_EXPORT Status Result::allocate(const daal::algorithms::PartialResult * partialResult, const Parameter * parameter, int method)
{
    const PartialResult * const partialRes = static_cast<const PartialResult *>(partialResult);

    Status s;
    if (method == normEqDense)
    {
        algorithmFPType dummy = 1.0;
        set(model, ridge_regression::ModelPtr(new ridge_regression::internal::ModelNormEqImpl(partialRes->getNumberOfFeatures(), partialRes->getNumberOfDependentVariables(),
                                                                *parameter, dummy, s)));
    }

    return s;
}

} // namespace training
} // namespace ridge_regression
} // namespace algorithms
} // namespace daal

#endif
