/* file: multiclassclassifier_train_kernel.h */
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
//  Declaration of template function that trains Multi-class slassifier model.
//--
*/

#ifndef __MULTICLASSCLASSIFIER_TRAIN_KERNEL_H__
#define __MULTICLASSCLASSIFIER_TRAIN_KERNEL_H__

#include "numeric_table.h"
#include "model.h"
#include "algorithm.h"
#include "multi_class_classifier_train_types.h"
#include "service_defines.h"

using namespace daal::data_management;

namespace daal
{
namespace algorithms
{
namespace multi_class_classifier
{
namespace training
{
namespace internal
{

template<Method method, typename AlgorithmFPtype, CpuType cpu>
struct MultiClassClassifierTrainKernel : public Kernel
{
    services::Status compute(const NumericTable *a0, const NumericTable *a1, daal::algorithms::Model *r,
                             const daal::algorithms::Parameter *par);
};

} // namespace internal

} // namespace training

} // namespace multi_class_classifier

} // namespace algorithms

} // namespace daal


#endif
