/* file: pca_onlineparameter_svd.h */
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
//  Implementation of PCA algorithm interface.
//--
*/

#ifndef __PCA_ONLINEPARAMETER_
#define __PCA_ONLINEPARAMETER_

#include "algorithms/pca/pca_types.h"

namespace daal
{
namespace algorithms
{
namespace pca
{
/** Constructs PCA parameters */
template <typename algorithmFPType>
OnlineParameter<algorithmFPType, svdDense>::OnlineParameter() {};

template <typename algorithmFPType>
services::Status OnlineParameter<algorithmFPType, svdDense>::check() const
{
    return services::Status();
}

} // namespace pca
} // namespace algorithms
} // namespace daal

#endif
