/* file: pca_batchparameter_svd_fpt.cpp */
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

#include "algorithms/pca/pca_types.h"

namespace daal
{
namespace algorithms
{
namespace pca
{
/** Constructs PCA parameters */
template <typename algorithmFPType>
DAAL_EXPORT BatchParameter<algorithmFPType, svdDense>::BatchParameter(const services::SharedPtr<normalization::zscore::BatchImpl> & normalization)
    : normalization(normalization) {};

template <typename algorithmFPType>
DAAL_EXPORT services::Status BatchParameter<algorithmFPType, svdDense>::check() const
{
    DAAL_CHECK(normalization, services::ErrorNullAuxiliaryAlgorithm);
    return services::Status();
}

template DAAL_EXPORT BatchParameter<DAAL_FPTYPE, svdDense>::BatchParameter(
    const services::SharedPtr<normalization::zscore::BatchImpl> & normalization);

template DAAL_EXPORT services::Status BatchParameter<DAAL_FPTYPE, svdDense>::check() const;

} // namespace pca
} // namespace algorithms
} // namespace daal
