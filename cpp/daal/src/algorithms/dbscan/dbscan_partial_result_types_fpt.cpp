/* file: dbscan_partial_result_types_fpt.cpp */
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
//  Implementation of DBSCAN algorithm and types methods.
//--
*/

#include "src/algorithms/dbscan/dbscan_partial_result.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace dbscan
{
template DAAL_EXPORT Status DistributedPartialResultStep1::allocate<DAAL_FPTYPE>(const daal::algorithms::Input * input,
                                                                                 const daal::algorithms::Parameter * parameter, const int method);
template DAAL_EXPORT Status DistributedPartialResultStep2::allocate<DAAL_FPTYPE>(const daal::algorithms::Input * input,
                                                                                 const daal::algorithms::Parameter * parameter, const int method);
template DAAL_EXPORT Status DistributedPartialResultStep3::allocate<DAAL_FPTYPE>(const daal::algorithms::Input * input,
                                                                                 const daal::algorithms::Parameter * parameter, const int method);
template DAAL_EXPORT Status DistributedPartialResultStep4::allocate<DAAL_FPTYPE>(const daal::algorithms::Input * input,
                                                                                 const daal::algorithms::Parameter * parameter, const int method);
template DAAL_EXPORT Status DistributedPartialResultStep5::allocate<DAAL_FPTYPE>(const daal::algorithms::Input * input,
                                                                                 const daal::algorithms::Parameter * parameter, const int method);
template DAAL_EXPORT Status DistributedPartialResultStep6::allocate<DAAL_FPTYPE>(const daal::algorithms::Input * input,
                                                                                 const daal::algorithms::Parameter * parameter, const int method);
template DAAL_EXPORT Status DistributedPartialResultStep7::allocate<DAAL_FPTYPE>(const daal::algorithms::Input * input,
                                                                                 const daal::algorithms::Parameter * parameter, const int method);
template DAAL_EXPORT Status DistributedPartialResultStep8::allocate<DAAL_FPTYPE>(const daal::algorithms::Input * input,
                                                                                 const daal::algorithms::Parameter * parameter, const int method);
template DAAL_EXPORT Status DistributedResultStep9::allocate<DAAL_FPTYPE>(const daal::algorithms::PartialResult * pres,
                                                                          const daal::algorithms::Parameter * parameter, const int method);
template DAAL_EXPORT Status DistributedPartialResultStep9::allocate<DAAL_FPTYPE>(const daal::algorithms::Input * input,
                                                                                 const daal::algorithms::Parameter * parameter, const int method);
template DAAL_EXPORT Status DistributedPartialResultStep10::allocate<DAAL_FPTYPE>(const daal::algorithms::Input * input,
                                                                                  const daal::algorithms::Parameter * parameter, const int method);
template DAAL_EXPORT Status DistributedPartialResultStep11::allocate<DAAL_FPTYPE>(const daal::algorithms::Input * input,
                                                                                  const daal::algorithms::Parameter * parameter, const int method);
template DAAL_EXPORT Status DistributedPartialResultStep12::allocate<DAAL_FPTYPE>(const daal::algorithms::Input * input,
                                                                                  const daal::algorithms::Parameter * parameter, const int method);
template DAAL_EXPORT Status DistributedResultStep13::allocate<DAAL_FPTYPE>(const daal::algorithms::PartialResult * pres,
                                                                           const daal::algorithms::Parameter * parameter, const int method);
template DAAL_EXPORT Status DistributedPartialResultStep13::allocate<DAAL_FPTYPE>(const daal::algorithms::Input * input,
                                                                                  const daal::algorithms::Parameter * parameter, const int method);

} // namespace dbscan
} // namespace algorithms
} // namespace daal
