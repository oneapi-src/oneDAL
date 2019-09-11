/* file: svm_predict_kernel.h */
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
//  Declaration of template structs that contains SVM prediction functions.
//--
*/

#ifndef __SVM_PREDICT_KERNEL_H__
#define __SVM_PREDICT_KERNEL_H__

#include "numeric_table.h"
#include "model.h"
#include "daal_defines.h"
#include "svm_predict_types.h"
#include "kernel.h"

using namespace daal::data_management;

namespace daal
{
namespace algorithms
{
namespace svm
{
namespace prediction
{
namespace internal
{

template <Method method, typename algorithmFPType, CpuType cpu>
struct SVMPredictImpl : public Kernel
{
    services::Status compute(const NumericTablePtr& xTable, const daal::algorithms::Model *m, NumericTable& r,
                             const daal::algorithms::Parameter *par);
};

} // namespace internal

} // namespace prediction

} // namespace svm

} // namespace algorithms

} // namespace daal

#endif
