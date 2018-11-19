/* file: logistic_regression_predict_result_fpt.cpp */
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
//  Implementation of the logistic regression algorithm interface
//--
*/

#include "algorithms/logistic_regression/logistic_regression_predict_types.h"

namespace daal
{
namespace algorithms
{
namespace logistic_regression
{
namespace prediction
{
namespace interface1
{

template<typename algorithmFPType>
DAAL_EXPORT services::Status Result::allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method)
{
    services::Status s;
    const logistic_regression::prediction::Parameter* prm = (const logistic_regression::prediction::Parameter*)parameter;
    const logistic_regression::prediction::Input* inp = static_cast<const logistic_regression::prediction::Input*>(input);
    const size_t nProb = (prm->nClasses == 2 ? 1 : prm->nClasses);
    if(prm->resultsToCompute & computeClassesLabels)
        s = classifier::prediction::Result::allocate<algorithmFPType>(input, parameter, method);
    if(s.ok() && (prm->resultsToCompute & computeClassesProbabilities))
        set(probabilities, data_management::HomogenNumericTable<algorithmFPType>::create(nProb,
            inp->getNumberOfRows(), data_management::NumericTableIface::doAllocate, &s));
    if(s.ok() && (prm->resultsToCompute & computeClassesLogProbabilities))
        set(logProbabilities, data_management::HomogenNumericTable<algorithmFPType>::create(nProb,
        inp->getNumberOfRows(), data_management::NumericTableIface::doAllocate, &s));
    return s;
}

template DAAL_EXPORT services::Status Result::allocate<DAAL_FPTYPE>(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method);

}
}// namespace prediction
}// namespace logistic_regression
}// namespace algorithms
}// namespace daal
