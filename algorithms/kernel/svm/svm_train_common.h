/* file: svm_train_common.h */
/*******************************************************************************
* Copyright 2020 Intel Corporation
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

#ifndef __SVM_TRAIN_COMMON_H__
#define __SVM_TRAIN_COMMON_H__

#include "service/kernel/data_management/service_numeric_table.h"
#include "externals/service_ittnotify.h"

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
using namespace daal::services::internal;

enum SVMVectorStatus
{
    free   = 0x0,
    up     = 0x1,
    low    = 0x2,
    shrink = 0x4
};

template <typename algorithmFPType, CpuType cpu>
struct HelperTrainSVM
{
    DAAL_FORCEINLINE static bool isUpper(const algorithmFPType y, const algorithmFPType alpha, const algorithmFPType C)
    {
        return (y > 0 && alpha < C) || (y < 0 && alpha > 0);
    }
    DAAL_FORCEINLINE static bool isLower(const algorithmFPType y, const algorithmFPType alpha, const algorithmFPType C)
    {
        return (y > 0 && alpha > 0) || (y < 0 && alpha < C);
    }

    DAAL_FORCEINLINE static algorithmFPType WSSi(size_t nActiveVectors, const algorithmFPType * grad, const char * I, int & Bi);

    DAAL_FORCEINLINE static void WSSjLocal(const size_t jStart, const size_t jEnd, const algorithmFPType * KiBlock,
                                           const algorithmFPType * kernelDiag, const algorithmFPType * grad, const char * I,
                                           const algorithmFPType GMin, const algorithmFPType Kii, const algorithmFPType tau, int & Bj,
                                           algorithmFPType & GMax, algorithmFPType & GMax2, algorithmFPType & delta);

private:
    DAAL_FORCEINLINE static void WSSjLocalBaseline(const size_t jStart, const size_t jEnd, const algorithmFPType * KiBlock,
                                                   const algorithmFPType * kernelDiag, const algorithmFPType * grad, const char * I,
                                                   const algorithmFPType GMin, const algorithmFPType Kii, const algorithmFPType tau, int & Bj,
                                                   algorithmFPType & GMax, algorithmFPType & GMax2, algorithmFPType & delta);
};

} // namespace internal
} // namespace training
} // namespace svm
} // namespace algorithms
} // namespace daal

#endif
