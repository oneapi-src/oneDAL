/* file: brownboost_train_kernel.h */
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
//  Declaration of template function that trains Brown Boost model.
//--
*/

#ifndef __BROWN_BOOST_TRAIN_KERNEL_H__
#define __BROWN_BOOST_TRAIN_KERNEL_H__

#include "brownboost_model.h"
#include "brownboost_training_types.h"
#include "kernel.h"
#include "service_numeric_table.h"

using namespace daal::data_management;

namespace daal
{
namespace algorithms
{
namespace brownboost
{
namespace training
{
namespace internal
{

template <Method method, typename algorithmFPType, CpuType cpu>
class BrownBoostTrainKernel : public Kernel
{
public:
    void compute(size_t n, NumericTablePtr *a, Model *r, const Parameter *par);

private:
    void updateWeights(size_t nVectors, algorithmFPType s, algorithmFPType c, algorithmFPType invSqrtC,
                       const algorithmFPType *r, algorithmFPType *nra, algorithmFPType *nre2, algorithmFPType *w);

    algorithmFPType *reallocateAlpha(size_t oldAlphaSize, size_t alphaSize, algorithmFPType *oldAlpha);

    void brownBoostFreundKernel(size_t nVectors,
                                NumericTablePtr weakLearnerInputTables[],
                                services::SharedPtr<daal::internal::HomogenNumericTableCPU<algorithmFPType, cpu> > hTable, algorithmFPType *y,
                                Model *boostModel, Parameter *parameter, size_t *nWeakLearnersPtr,
                                algorithmFPType **alphaPtr);
};

template <Method method, typename algorithmFPType, CpuType cpu>
struct NewtonRaphsonKernel
{
    NewtonRaphsonKernel(size_t nVectors, Parameter *parameter,
                        services::SharedPtr<services::KernelErrorCollection> _errors);

    virtual ~NewtonRaphsonKernel();

    void compute(algorithmFPType gamma, algorithmFPType s, algorithmFPType *h, algorithmFPType *y);

    size_t nVectors;
    algorithmFPType nrT;
    algorithmFPType nrAlpha;

    algorithmFPType *nrd;
    algorithmFPType *nrw;
    algorithmFPType *nra;
    algorithmFPType *nrb;
    algorithmFPType *nre1;
    algorithmFPType *nre2;

    size_t nrMaxIter;
    algorithmFPType error;
    algorithmFPType nrAccuracy;
    algorithmFPType nu;
    algorithmFPType c;
    algorithmFPType invC;
    algorithmFPType sqrtC;
    algorithmFPType invSqrtC;
    algorithmFPType sqrtPiC;

    services::SharedPtr<services::KernelErrorCollection> _errors;
};

} // namespace daal::algorithms::brownboost::training::internal
}
}
}
} // namespace daal

#endif
