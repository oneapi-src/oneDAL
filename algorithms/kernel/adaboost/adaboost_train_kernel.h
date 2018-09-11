/* file: adaboost_train_kernel.h */
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
//  Declaration of template function that trains Ada Boost model.
//--
*/

#ifndef __ADABOOST_TRAIN_KERNEL_H__
#define __ADABOOST_TRAIN_KERNEL_H__

#include "adaboost_model.h"
#include "adaboost_training_types.h"
#include "kernel.h"
#include "service_numeric_table.h"

using namespace daal::data_management;

namespace daal
{
namespace algorithms
{
namespace adaboost
{
namespace training
{
namespace internal
{

template <Method method, typename algorithmFPType, CpuType cpu>
class AdaBoostTrainKernel : public Kernel
{
public:
    services::Status compute(size_t n, NumericTablePtr *a, Model *r, const Parameter *par);
    typedef typename daal::internal::HomogenNumericTableCPU<algorithmFPType, cpu> HomogenNT;
    typedef typename services::SharedPtr<HomogenNT> HomogenNTPtr;

private:
    services::Status adaBoostFreundKernel(size_t nVectors, NumericTablePtr weakLearnerInputTables[], const HomogenNTPtr& hTable,
        const algorithmFPType *y, Model *boostModel, Parameter *parameter, size_t& nWeakLearners, algorithmFPType *alpha);
};

} // namespace daal::algorithms::adaboost::training::internal
}
}
}
} // namespace daal

#endif
