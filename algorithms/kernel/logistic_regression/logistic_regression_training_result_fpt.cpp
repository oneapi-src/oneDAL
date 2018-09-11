/* file: logistic_regression_training_result_fpt.cpp */
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

#include "algorithms/logistic_regression/logistic_regression_training_types.h"
#include "logistic_regression_model_impl.h"

namespace daal
{
namespace algorithms
{
namespace logistic_regression
{

namespace internal
{
template <typename modelFPType>
ModelImpl::ModelImpl(size_t nFeatures, bool interceptFlag, size_t nClasses, modelFPType dummy, services::Status* st) :
    ClassificationImplType(nFeatures), _interceptFlag(interceptFlag)
{
    const size_t nRows = nClasses == 2 ? 1 : nClasses;
    const size_t nCols = nFeatures + 1;
    _beta = data_management::HomogenNumericTable<modelFPType>::create(nCols, nRows, data_management::NumericTable::doAllocate, 0, st);
}
}

namespace training
{

template<typename algorithmFPType>
DAAL_EXPORT services::Status Result::allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method)
{
    services::Status s;
    const classifier::training::Input* inp = static_cast<const classifier::training::Input*>(input);
    const size_t nFeatures = inp->get(classifier::training::data)->getNumberOfColumns();
    const logistic_regression::training::Parameter* prm = (const logistic_regression::training::Parameter*)parameter;
    set(classifier::training::model, ModelPtr(new logistic_regression::internal::ModelImpl(
        nFeatures, prm->interceptFlag, prm->nClasses, algorithmFPType(0), &s)));
    return s;
}

template DAAL_EXPORT services::Status Result::allocate<DAAL_FPTYPE>(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method);

}// namespace training
}// namespace logistic_regression
}// namespace algorithms
}// namespace daal
