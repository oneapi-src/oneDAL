/* file: pca_onlineparameter_correlation_fpt.cpp */
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
//  Implementation of PCA algorithm interface.
//--
*/

#include "pca_onlineparameter_correlation.h"

namespace daal
{
namespace algorithms
{
namespace pca
{
/**
 * Constructs the ridge regression model
 * \param[in] featnum Number of features in the training data
 * \param[in] nrhs    Number of responses in the training data
 * \param[in] par     Ridge regression parameters
 * \param[in] dummy   Dummy variable for the templated constructor
 */

template DAAL_EXPORT OnlineParameter<DAAL_FPTYPE, correlationDense>::OnlineParameter(const services::SharedPtr<covariance::OnlineImpl> & covariance);
template DAAL_EXPORT services::Status OnlineParameter<DAAL_FPTYPE, correlationDense>::check() const;

} // namespace pca
} // namespace algorithms
} // namespace daal
