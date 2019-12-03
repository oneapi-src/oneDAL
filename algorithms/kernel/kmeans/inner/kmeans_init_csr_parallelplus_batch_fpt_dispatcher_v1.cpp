/* file: kmeans_init_csr_parallelplus_batch_fpt_dispatcher_v1.cpp */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation
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
//  Implementation of K-means++ initialization algorithm container
//--
*/

#include "kmeans/inner/kmeans_init_container_v1.h"

namespace daal
{
namespace algorithms
{
__DAAL_INSTANTIATE_DISPATCH_CONTAINER_KM(kmeans::init::interface1::BatchContainer, batch, DAAL_FPTYPE, kmeans::init::parallelPlusCSR)
} // namespace algorithms
} // namespace daal
