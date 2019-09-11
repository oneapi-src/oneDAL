/* file: linear_regression_training_result.h */
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
//  Implementation of the linear regression algorithm interface
//--
*/

#ifndef __LINEAR_REGRESSION_TRAINING_RESULT_
#define __LINEAR_REGRESSION_TRAINING_RESULT_

#include "algorithms/linear_regression/linear_regression_training_types.h"
#include "linear_regression_ne_model_impl.h"
#include "linear_regression_qr_model_impl.h"

namespace daal
{
namespace algorithms
{
namespace linear_regression
{
namespace training
{
using namespace daal::services;

/**
 * Allocates memory to store the result of linear regression model-based training
 * \param[in] input Pointer to an object containing the input data
 * \param[in] method Computation method for the algorithm
 * \param[in] parameter %Parameter of linear regression model-based training
 */
template<typename algorithmFPType>
DAAL_EXPORT Status Result::allocate(const daal::algorithms::Input *input, const Parameter *parameter, const int method)
{
    const Input *in = static_cast<const Input *>(input);

    Status s;
    const algorithmFPType dummy = 1.0;
    if(method == qrDense)
    {
        set(model, linear_regression::ModelPtr(new linear_regression::internal::ModelQRImpl(in->getNumberOfFeatures(), in->getNumberOfDependentVariables(), *parameter, dummy, s)));
    }
    else if(method == normEqDense)
    {
        set(model, linear_regression::ModelPtr(new linear_regression::internal::ModelNormEqImpl(in->getNumberOfFeatures(), in->getNumberOfDependentVariables(), *parameter, dummy, s)));
    }

    return s;
}

/**
 * Allocates memory to store the result of linear regression model-based training
 * \param[in] partialResult Pointer to an object containing the input data
 * \param[in] method        Computation method of the algorithm
 * \param[in] parameter     %Parameter of linear regression model-based training
 */
template<typename algorithmFPType>
DAAL_EXPORT Status Result::allocate(const daal::algorithms::PartialResult *partialResult, const Parameter *parameter, const int method)
{
    const PartialResult *partialRes = static_cast<const PartialResult *>(partialResult);

    Status s;
    const algorithmFPType dummy = 1.0;
    if(method == qrDense)
    {
        set(model, linear_regression::ModelPtr(new linear_regression::internal::ModelQRImpl(partialRes->getNumberOfFeatures(), partialRes->getNumberOfDependentVariables(), *parameter, dummy, s)));
    }
    else if(method == normEqDense)
    {
        set(model, linear_regression::ModelPtr(new linear_regression::internal::ModelNormEqImpl(partialRes->getNumberOfFeatures(), partialRes->getNumberOfDependentVariables(), *parameter, dummy, s)));
    }

    return s;
}

} // namespace training
} // namespace linear_regression
} // namespace algorithms
} // namespace daal

#endif
