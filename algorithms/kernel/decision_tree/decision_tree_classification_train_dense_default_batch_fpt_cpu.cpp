/* file: decision_tree_classification_train_dense_default_batch_fpt_cpu.cpp */
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
//  Implementation of Decision tree training functions for the defaultDense method.
//--
*/

#include "decision_tree_classification_train_container.h"
#include "decision_tree_classification_train_dense_default_impl.i"

namespace daal
{
namespace algorithms
{
namespace decision_tree
{
namespace classification
{
namespace training
{
namespace interface1
{

template class BatchContainer<DAAL_FPTYPE, defaultDense, DAAL_CPU>;

} // namespace interface1

namespace internal
{

template class DecisionTreeTrainBatchKernel<DAAL_FPTYPE, defaultDense, DAAL_CPU>;

} // namespace internal
} // namespace training
} // namespace classification
} // namespace decision_tree
} // namespace algorithms
} // namespace daal
