/* file: kmeans_init_csr_parallelplus_batch_fpt_cpu.cpp */
/*******************************************************************************
* Copyright 2014 Intel Corporation
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
//  Implementation of parallel K-means++ initialization method for K-means algorithm
//--
*/

#include "src/algorithms/kmeans/kmeans_init_kernel.h"
#include "src/algorithms/kmeans/kmeans_plusplus_init_impl.i"
#include "src/algorithms/kmeans/kmeans_init_container.h"

namespace daal
{
namespace algorithms
{
namespace kmeans
{
namespace init
{
namespace interface2
{
template class BatchContainer<DAAL_FPTYPE, parallelPlusCSR, DAAL_CPU>;
}
namespace internal
{
template class KMeansInitKernel<parallelPlusCSR, DAAL_FPTYPE, DAAL_CPU>;
} // namespace internal
} // namespace init
} // namespace kmeans
} // namespace algorithms
} // namespace daal
