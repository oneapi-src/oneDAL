/* file: svm_train_boser_batch_fpt_cpu.cpp */
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
//  Implementation of SVM boser training algorithm.
//--
*/

#include "svm_train_batch_container.h"
#include "svm_train_boser_kernel.h"
#include "svm_train_boser_impl.i"

namespace daal
{
namespace algorithms
{
namespace svm
{
namespace training
{
namespace interface1
{
template class BatchContainer<DAAL_FPTYPE, boser, DAAL_CPU>;
}
namespace internal
{

template struct SVMTrainImpl<boser, DAAL_FPTYPE, DAAL_CPU>;

} // namespace internal

} // namespace training

} // namespace svm

} // namespace algorithms

} // namespace daal
