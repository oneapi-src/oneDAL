/* file: logistic_regression_train_dense_default_batch_fpt_dispatcher.cpp */
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
//  Implementation of logistic regression container.
//--
*/

#include "logistic_regression_train_container.h"
#include "daal_strings.h"

namespace daal
{
namespace algorithms
{
namespace interface1
{
__DAAL_INSTANTIATE_DISPATCH_CONTAINER(logistic_regression::training::BatchContainer, batch, DAAL_FPTYPE, \
    logistic_regression::training::defaultDense)
}
namespace logistic_regression
{
namespace training
{
namespace interface1
{
template <>
Batch<DAAL_FPTYPE, logistic_regression::training::defaultDense>::Batch(size_t nClasses, const SolverPtr& solver)
{
    _par = new ParameterType(nClasses, solver);
    initialize();
}

using BatchType = Batch<DAAL_FPTYPE, logistic_regression::training::defaultDense>;
template <>
Batch<DAAL_FPTYPE, logistic_regression::training::defaultDense>::Batch(const BatchType &other): classifier::training::Batch(other), input(other.input)
{
    _par = new ParameterType(other.parameter());
    initialize();
}

}
}
}
}
} // namespace daal
