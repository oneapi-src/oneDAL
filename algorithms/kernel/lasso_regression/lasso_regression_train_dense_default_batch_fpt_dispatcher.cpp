/* file: lasso_regression_train_dense_default_batch_fpt_dispatcher.cpp */
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
//  Implementation of lasso regression container.
//--
*/

#include "lasso_regression_train_container.h"

#include "daal_strings.h"

namespace daal
{
namespace algorithms
{
__DAAL_INSTANTIATE_DISPATCH_CONTAINER(lasso_regression::training::BatchContainer, batch, DAAL_FPTYPE, \
    lasso_regression::training::defaultDense)

namespace lasso_regression
{
namespace training
{
namespace interface1
{
template <>
Batch<DAAL_FPTYPE, lasso_regression::training::defaultDense>::Batch(const optimization_solver::iterative_solver::BatchPtr& solver)
{
    _par = new ParameterType(solver);
    initialize();
}

using BatchType = Batch<DAAL_FPTYPE, lasso_regression::training::defaultDense>;
template <>
Batch<DAAL_FPTYPE, lasso_regression::training::defaultDense>::Batch(const BatchType &other): input(other.input)
{
    _par = new ParameterType(other.parameter());
    initialize();
}

}
}
}
}
} // namespace daal
