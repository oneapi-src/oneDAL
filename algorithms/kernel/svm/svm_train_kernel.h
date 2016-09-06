/* file: svm_train_kernel.h */
/*******************************************************************************
* Copyright 2014-2016 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

/*
//++
//  Declaration of template structs that calculate SVM Training functions.
//--
*/

#ifndef __SVM_TRAIN_KERNEL_H__
#define __SVM_TRAIN_KERNEL_H__

#include "numeric_table.h"
#include "model.h"
#include "daal_defines.h"
#include "svm_train_types.h"
#include "kernel.h"
#include "service_micro_table.h"

using namespace daal::data_management;
using namespace daal::internal;

#include "svm_train_boser_cache.i"

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

enum SVMVectorStatus
{
    free    = 0x0,
    up      = 0x1,
    low     = 0x2,
    shrink  = 0x4
};

template <typename algorithmFPType, CpuType cpu>
struct SVMTrainTask
{
    SVMTrainTask(size_t cacheSize, size_t nVectors, size_t kernelFunctionBlockSize, bool doShrinking,
                 NumericTablePtr xTable, NumericTable *yTable,
                 services::SharedPtr<kernel_function::KernelIface> kernel,
                 services::SharedPtr<services::KernelErrorCollection> _errors);

    virtual ~SVMTrainTask();

    void init(algorithmFPType C);

    inline void updateI(algorithmFPType C, size_t index);

    size_t nVectors;
    algorithmFPType *y;
    algorithmFPType *alpha;
    algorithmFPType *grad;
    algorithmFPType *kernelDiag;
    char *I;

    SVMCacheIface<algorithmFPType, cpu> *cache;
    services::SharedPtr<services::KernelErrorCollection> _errors;
};

template <Method method, typename algorithmFPType, CpuType cpu>
struct SVMTrainImpl : public Kernel
{
    void compute(NumericTablePtr xTable, NumericTable *yTable, daal::algorithms::Model *r,
                 const daal::algorithms::Parameter *par);
};

} // namespace internal

} // namespace training

} // namespace svm

} // namespace algorithms

} // namespace daal

#endif
