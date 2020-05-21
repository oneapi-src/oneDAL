/* file: communicator_oneccl_sycl.h */
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

#ifndef __DAAL_ONEAPI_INTERNAL_COMMUNICATOR_ONECCL_SYCL_H__
#define __DAAL_ONEAPI_INTERNAL_COMMUNICATOR_ONECCL_SYCL_H__

#include <CL/cl.h>
#include <CL/sycl.hpp>
#include "ccl.h"
#include "oneapi/internal/communicator.h"
#include "oneapi/internal/types_comm_utils_cxx11.h"

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
 *  <a name="DAAL-CLASS-ONEAPI-INTERNAL__COMMUNICATORONECCLIMPL"></a>
 *  \brief Implementation of oneCCL-based communicator
 */
class CommunicatorOneCclImpl : public CommunicatorIface
{
public:
    CommunicatorOneCclImpl(cl::sycl::queue & deviceQueue) : _deviceQueue(deviceQueue)
    {
        ccl_init();
        ccl_get_comm_rank(NULL, &_rank);
        ccl_get_comm_size(NULL, &_size);
        ccl_stream_create(ccl_stream_sycl, &deviceQueue, &_stream);
    }
    ~CommunicatorOneCclImpl() { ccl_finalize(); }

    void allReduceSum(oneapi::internal::UniversalBuffer src, oneapi::internal::UniversalBuffer dest, size_t count,
                      daal::services::Status * status = nullptr) DAAL_C11_OVERRIDE
    {
        DAAL_ASSERT(dest.type() == src.type());
        try
        {
            BufferAllReducer::allReduceSum(_deviceQueue, _stream, dest, src, count);
        }
        catch (cl::sycl::exception const & e)
        {
            oneapi::internal::convertSyclExceptionToStatus(e, status);
        }
    }
    void allGatherV(oneapi::internal::UniversalBuffer dest, size_t * recvCount, oneapi::internal::UniversalBuffer src, size_t srcCount,
                    daal::services::Status * status = nullptr) DAAL_C11_OVERRIDE
    {
        DAAL_ASSERT(dest.type() == src.type());
        try
        {
            BufferAllGatherer::allGatherV(_deviceQueue, _stream, dest, recvCount, src, srcCount);
        }
        catch (cl::sycl::exception const & e)
        {
            oneapi::internal::convertSyclExceptionToStatus(e, status);
        }
    }
    size_t size() const DAAL_C11_OVERRIDE { return _size; }
    size_t rank() const DAAL_C11_OVERRIDE { return _rank; }

private:
    cl::sycl::queue _deviceQueue;
    ccl_stream_t _stream;
    size_t _rank;
    size_t _size;
};

/** } */
} // namespace interface1

using interface1::CommunicatorOneCclImpl;

} // namespace internal
} // namespace comm
} // namespace preview
} // namespace daal

#endif // DAAL_SYCL_INTERFACE
