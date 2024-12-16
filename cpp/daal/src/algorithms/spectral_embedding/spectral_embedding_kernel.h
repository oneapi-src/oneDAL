/* file: spectral_embedding_kernel.h */
/*******************************************************************************
* Copyright contributors to the oneDAL project
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

#ifndef __SPECTRAL_EMBEDDING_KERNEL_H__
#define __SPECTRAL_EMBEDDING_KERNEL_H__

#include "data_management/data/numeric_table.h"
#include "services/daal_defines.h"
#include "src/algorithms/kernel.h"

namespace daal
{
namespace algorithms
{
namespace spectral_embedding
{

enum Method
{
    defaultDense = 0
};

namespace internal
{

using namespace daal::data_management;
using namespace daal::services;

struct KernelParameter : daal::algorithms::Parameter
{
    size_t numberOfEmbeddings = 1;
    size_t numberOfNeighbors  = 1;
};

template <typename algorithmFPType, Method method, CpuType cpu>
struct SpectralEmbeddingKernel : public Kernel
{
    services::Status compute(const NumericTable * xTable, NumericTable * embeddingTable, NumericTable * eigenTable, const KernelParameter & par);
};

} // namespace internal
} // namespace spectral_embedding
} // namespace algorithms
} // namespace daal

#endif
