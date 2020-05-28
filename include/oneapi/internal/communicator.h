/* file: communicator.h */
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

#ifndef __DAAL_ONEAPI_INTERNAL_COMMUNICATOR_H__
#define __DAAL_ONEAPI_INTERNAL_COMMUNICATOR_H__

#include "services/buffer.h"
#include "services/error_handling.h"
#include "services/internal/error_handling_helpers.h"
#include "services/internal/any.h"
#include "oneapi/internal/types.h"

namespace daal
{
namespace preview
{
namespace comm
{
namespace internal
{
namespace interface1
{
/**
 *  <a name="DAAL-CLASS-ONEAPI-INTERNAL__COMMUNICATORIFACE"></a>
 *  \brief Interface for inter-process communicator
 */
class CommunicatorIface
{
public:
    virtual ~CommunicatorIface() {}
    virtual void allReduceSum(oneapi::internal::UniversalBuffer dest, oneapi::internal::UniversalBuffer src, size_t count,
                              daal::services::Status * status = nullptr) = 0;
    virtual void allGatherV(oneapi::internal::UniversalBuffer dest, size_t * recvCount, oneapi::internal::UniversalBuffer src, size_t srcCount,
                            daal::services::Status * status = nullptr)   = 0;
    virtual size_t size() const                                          = 0;
    virtual size_t rank() const                                          = 0;
};

/**
 *  <a name="DAAL-CLASS-ONEAPI-INTERNAL__NOCOMMUNICATOR"></a>
 *  \brief Communicator stub for a serial run
 */
class DummyCommunicator : public CommunicatorIface
{
public:
    virtual void allReduceSum(oneapi::internal::UniversalBuffer dest, oneapi::internal::UniversalBuffer src, size_t count,
                              daal::services::Status * status = nullptr)
    {}
    virtual void allGatherV(oneapi::internal::UniversalBuffer dest, size_t * recvCount, oneapi::internal::UniversalBuffer src, size_t srcCount,
                            daal::services::Status * status = nullptr)
    {}
    virtual size_t size() const { return 1; }
    virtual size_t rank() const { return 0; }
};

/** } */
} // namespace interface1

using interface1::CommunicatorIface;
using interface1::DummyCommunicator;
} // namespace internal
} // namespace comm
} // namespace preview
} // namespace daal

#endif // DAAL_SYCL_INTERFACE
