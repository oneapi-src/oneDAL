/* file: qr_dense_default_distr_step3_fpt.cpp */
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
//  Implementation of qr algorithm and types methods.
//--
*/

#include "src/algorithms/qr/qr_dense_default_distr_step3.h"
namespace daal
{
namespace algorithms
{
namespace qr
{
namespace interface1
{
template DAAL_EXPORT services::Status DistributedPartialResultStep3::allocate<DAAL_FPTYPE>(const daal::algorithms::Input * input,
                                                                                           const daal::algorithms::Parameter * parameter,
                                                                                           const int method);
template DAAL_EXPORT services::Status DistributedPartialResultStep3::setPartialResultStorage<DAAL_FPTYPE>(
    data_management::DataCollection * qCollection);

} // namespace interface1
} // namespace qr
} // namespace algorithms
} // namespace daal
