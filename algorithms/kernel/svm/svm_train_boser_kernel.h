/* file: svm_train_boser_kernel.h */
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
    static const size_t kernelFunctionBlockSize = 1024; /* Size of the block of kernel function elements */

    void compute(NumericTablePtr xTable, NumericTable *yTable, daal::algorithms::Model *r,
                 const daal::algorithms::Parameter *par);

protected:
    /**
     * \brief Write support vectors and classification coefficients into output model
     */
    size_t computeNumberOfSV(size_t nVectors, const algorithmFPType *alpha);
    void setSVCoefficients(size_t nVectors, size_t nSV, const algorithmFPType *y, const algorithmFPType *alpha,
                Model *model);

    void setSV(Model *model, BlockMicroTable<algorithmFPType, readOnly, cpu> &mtX,
                size_t nFeatures, size_t nVectors, size_t nSV, const algorithmFPType *alpha,
                SVMCacheIface<algorithmFPType, cpu> *cache);

    void setSV(Model *model, CSRBlockMicroTable<algorithmFPType, readOnly, cpu> &mtX,
                size_t nFeatures, size_t nVectors, size_t nSV, const algorithmFPType *alpha,
                SVMCacheIface<algorithmFPType, cpu> *cache);

    algorithmFPType calculateBias(algorithmFPType C, size_t nVectors, const algorithmFPType *y,
                const algorithmFPType *alpha, const algorithmFPType *grad);

    bool findMaximumViolatingPair(size_t nActiveVectors, algorithmFPType tau, const algorithmFPType *y,
                const algorithmFPType *grad, const algorithmFPType *kernelDiag, char *I,
                SVMCacheIface<algorithmFPType, cpu> *cache, int *BiPtr, int *BjPtr,
                algorithmFPType *deltaPtr, algorithmFPType *maPtr, algorithmFPType *MaPtr,
                algorithmFPType *curEps);

    algorithmFPType WSSi(size_t nActiveVectors, const algorithmFPType *y, const algorithmFPType *grad,
                const algorithmFPType *kernelDiag, char *I, int *BiPtr);

    algorithmFPType WSSj(size_t nActiveVectors, algorithmFPType tau, const algorithmFPType *y,
                const algorithmFPType *grad, const algorithmFPType *kernelDiag, char *I,
                int Bi, SVMCacheIface<algorithmFPType, cpu> *cache, algorithmFPType GMax, int *BjPtr,
                algorithmFPType *deltaPtr);

    void updateTask(size_t nActiveVectors, algorithmFPType C, int Bi, int Bj, algorithmFPType delta, const algorithmFPType *y,
                    algorithmFPType *alpha, algorithmFPType *grad, SVMTrainTask<algorithmFPType, cpu> &task);

    inline void updateAlpha(algorithmFPType C, int Bi, int Bj, algorithmFPType delta, const algorithmFPType *y,
                algorithmFPType *alpha, algorithmFPType *newDeltai, algorithmFPType *newDeltaj);

    /*** Methods used in shrinking ***/
    size_t shrinkTask(size_t nActiveVectors, algorithmFPType *y, algorithmFPType *alpha, algorithmFPType *grad,
                algorithmFPType *kernelDiag, char *I);

    size_t updateShrinkingFlags(size_t nActiveVectors, algorithmFPType C, algorithmFPType ma, algorithmFPType Ma,
                const algorithmFPType *y, const algorithmFPType *alpha, const algorithmFPType *grad, char *I);

    void computeOptimalityConditionValues(size_t nVectors, char *I, const algorithmFPType *y,
                const algorithmFPType *grad, algorithmFPType *maPtr, algorithmFPType *MaPtr);

    size_t reconstructGradient(size_t nVectors, size_t nActiveVectors, SVMCacheIface<algorithmFPType, cpu> *cache,
                const algorithmFPType *y, const algorithmFPType *alpha, algorithmFPType *grad);
};


} // namespace internal

} // namespace training

} // namespace svm

} // namespace algorithms

} // namespace daal

#endif
