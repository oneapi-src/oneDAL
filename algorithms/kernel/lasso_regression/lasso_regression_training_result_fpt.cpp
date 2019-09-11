/* file: lasso_regression_training_result_fpt.cpp */
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
//  Implementation of the lasso regression algorithm interface
//--
*/

#include "algorithms/lasso_regression/lasso_regression_training_types.h"
#include "lasso_regression_model_impl.h"

namespace daal
{
namespace algorithms
{
namespace lasso_regression
{

namespace training
{
using namespace daal::services;
/**
 * Allocates memory to store the result of lasso regression model-based training
 * \param[in] input Pointer to an object containing the input data
 * \param[in] parameter %Parameter of lasso regression model-based training
 * \param[in] method Computation method for the algorithm
 */
template<typename algorithmFPType>
DAAL_EXPORT Status Result::allocate(const daal::algorithms::Input * input, const Parameter * parameter, const int method)
{
    const Input * const in = static_cast<const Input *>(input);

    Status s;
    const algorithmFPType dummy = 1.0;
    lasso_regression::internal::ModelImpl* mImpl = new lasso_regression::internal::ModelImpl(in->getNumberOfFeatures(),
                                                                                    in->getNumberOfDependentVariables(),
                                                                                    *parameter, dummy, s);
    set(model, lasso_regression::ModelPtr(mImpl));

    if(parameter->optResultToCompute & computeGramMatrix)
        set(gramMatrixId, data_management::HomogenNumericTable<algorithmFPType>::create(in->getNumberOfFeatures(),
            in->getNumberOfFeatures(), data_management::NumericTableIface::doAllocate, &s));
    return s;
}

template DAAL_EXPORT services::Status Result::allocate<DAAL_FPTYPE>(const daal::algorithms::Input *input, const Parameter *parameter, const int method);

}// namespace training
}// namespace lasso_regression
}// namespace algorithms
}// namespace daal
