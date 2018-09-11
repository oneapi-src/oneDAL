/* file: ridge_regression_training_partialresult.h */
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

#ifndef __RIDGE_REGRESSION_TRAINING_PARTIALRESULT_
#define __RIDGE_REGRESSION_TRAINING_PARTIALRESULT_

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

/**
 * Allocates memory to store a partial result of ridge regression model-based training
 * \param[in] input %Input object for the algorithm
 * \param[in] method Method of ridge regression model-based training
 * \param[in] parameter %Parameter of ridge regression model-based training
 */
template <typename algorithmFPType>
DAAL_EXPORT services::Status PartialResult::allocate(const daal::algorithms::Input * input, const daal::algorithms::Parameter * parameter, const int method)
{
    services::Status s;
    if (method == normEqDense)
    {
        /* input object can be an instance of both Input and DistributedInput<step2Master> classes.
           Both classes have multiple inheritance with InputIface as a second base class.
           That's why we use dynamic_cast here. */
        const InputIface * const in = dynamic_cast<const InputIface *>(input);
        const Parameter &par = *(static_cast<const Parameter *>(parameter));
        const algorithmFPType dummy = 1.0;
        set(partialModel, ridge_regression::ModelPtr(new ridge_regression::internal::ModelNormEqImpl(in->getNumberOfFeatures(), in->getNumberOfDependentVariables(), par, dummy, s)));
    }

    return s;
}

template <typename algorithmFPType>
DAAL_EXPORT services::Status PartialResult::initialize(const daal::algorithms::Input * input, const daal::algorithms::Parameter * parameter, const int method)
{
    return get(partialModel)->initialize();
}

} // namespace training
} // namespace ridge_regression
} // namespace algorithms
} // namespace daal

#endif
