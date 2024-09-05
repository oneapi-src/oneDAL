/* file: pca_partialresult_svd_fpt.cpp */
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

#include "src/algorithms/pca/pca_partialresult_svd.h"

namespace daal
{
namespace algorithms
{
namespace pca
{
template DAAL_EXPORT services::Status PartialResult<svdDense>::allocate<DAAL_FPTYPE>(const daal::algorithms::Input * input,
                                                                                     const daal::algorithms::Parameter * parameter, const int method);
template DAAL_EXPORT services::Status PartialResult<svdDense>::initialize<DAAL_FPTYPE>(const daal::algorithms::Input * input,
                                                                                       const daal::algorithms::Parameter * parameter,
                                                                                       const int method);

} // namespace pca
} // namespace algorithms
} // namespace daal
