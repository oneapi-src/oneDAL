/* file: ridge_regression_train_dense_normeq_batch_fpt_cpu.cpp */
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
//  Implementation of ridge regression training functions for the method
//  of normal equations.
//--
*/

#include "ridge_regression_train_container.h"
#include "ridge_regression_train_dense_normeq_impl.i"

namespace daal
{
namespace algorithms
{
namespace ridge_regression
{
namespace training
{
namespace interface1
{

template class BatchContainer<DAAL_FPTYPE, normEqDense, DAAL_CPU>;

} // namespace interface1

namespace internal
{

template class BatchKernel<DAAL_FPTYPE, normEqDense, DAAL_CPU>;

} // namespace internal
} // namespace training
} // namespace ridge_regression
} // namespace algorithms
} // namespace daal
