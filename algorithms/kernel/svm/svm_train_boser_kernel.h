/* file: svm_train_boser_kernel.h */
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
//  Declaration of template structs that calculate SVM Training functions.
//--
*/

#ifndef __SVM_TRAIN_BOSER_KERNEL_H__
#define __SVM_TRAIN_BOSER_KERNEL_H__

#include "numeric_table.h"
#include "model.h"
#include "daal_defines.h"
#include "svm_train_types.h"
#include "kernel.h"
#include "service_micro_table.h"

using namespace daal::data_management;
using namespace daal::internal;

#include "svm_train_kernel.h"

namespace daal
{
namespace algorithms
{
namespace svm
{
namespace training
{
namespace internal
{

template <typename algorithmFPType, CpuType cpu>
struct SVMTrainImpl<boser, algorithmFPType, cpu> : public Kernel
{
    services::Status compute(const NumericTablePtr& xTable, NumericTable& yTable, daal::algorithms::Model *r,
                             const daal::algorithms::Parameter *par);
};


} // namespace internal

} // namespace training

} // namespace svm

} // namespace algorithms

} // namespace daal

#endif
