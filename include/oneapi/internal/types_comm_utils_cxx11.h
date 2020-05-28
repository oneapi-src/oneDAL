/* file: types_comm_utils_cxx11.h */
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

#ifndef __DAAL_ONEAPI_INTERNAL_TYPES_COMM_UTILS_CXX11_H__
#define __DAAL_ONEAPI_INTERNAL_TYPES_COMM_UTILS_CXX11_H__

#include "oneapi/internal/types.h"
#include "oneapi/internal/types_utils.h"
#include "ccl.h"
#include "ccl_types.h"

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
template <typename T>
inline ccl_datatype_t get_ccl_datatype()
{
    return ccl_dtype_last_value;
}

#define DAAL_DECLARE_CCL_TYPE_ID_MAP(T_, ccl_dtype_) \
    template <>                                      \
    inline ccl_datatype_t get_ccl_datatype<T_>()     \
    {                                                \
        return ccl_dtype_;                           \
    }

DAAL_DECLARE_CCL_TYPE_ID_MAP(char, ccl_dtype_char)
DAAL_DECLARE_CCL_TYPE_ID_MAP(int, ccl_dtype_int)
DAAL_DECLARE_CCL_TYPE_ID_MAP(float, ccl_dtype_float)
DAAL_DECLARE_CCL_TYPE_ID_MAP(double, ccl_dtype_double)

/**
 *  <a name="DAAL-CLASS-ONEAPI-INTERNAL__BUFFERCOPIER"></a>
 *  \brief AllReducer for two UniversalBuffers
 */
class BufferAllReducer
{
private:
    struct Execute
    {
        cl::sycl::queue & queue;
        ccl_stream_t & stream;
        oneapi::internal::UniversalBuffer & dstUnivers;
        oneapi::internal::UniversalBuffer & srcUnivers;
        size_t count;

        explicit Execute(cl::sycl::queue & queue, ccl_stream_t & stream, oneapi::internal::UniversalBuffer & dst,
                         oneapi::internal::UniversalBuffer & src, size_t count)
            : queue(queue), stream(stream), dstUnivers(dst), srcUnivers(src), count(count)
        {}

        template <typename T>
        void operator()(oneapi::internal::Typelist<T>)
        {
            auto src = srcUnivers.get<T>().toSycl();
            auto dst = dstUnivers.get<T>().toSycl();
            ccl_request_t request;
            ccl_allreduce(&src, &dst, count, get_ccl_datatype<T>(), ccl_reduction_sum, NULL, NULL, stream, &request);
            ccl_wait(request);
        }
    };

public:
    static void allReduceSum(cl::sycl::queue & queue, ccl_stream_t & stream, oneapi::internal::UniversalBuffer & dst,
                             oneapi::internal::UniversalBuffer & src, size_t count)
    {
        Execute op(queue, stream, dst, src, count);
        oneapi::internal::TypeDispatcher::dispatch(dst.type(), op);
    }
};

/**
 *  <a name="DAAL-CLASS-ONEAPI-INTERNAL__BUFFERCOPIER"></a>
 *  \brief AllGatherV for two UniversalBuffers
 */
class BufferAllGatherer
{
private:
    struct Execute
    {
        cl::sycl::queue & queue;
        ccl_stream_t & stream;
        oneapi::internal::UniversalBuffer & dstUnivers;
        oneapi::internal::UniversalBuffer & srcUnivers;
        size_t srcCount;
        size_t * recvCount;

        explicit Execute(cl::sycl::queue & queue, ccl_stream_t & stream, oneapi::internal::UniversalBuffer & dst, size_t * recvCount,
                         oneapi::internal::UniversalBuffer & src, size_t srcCount)
            : queue(queue), stream(stream), dstUnivers(dst), srcUnivers(src), recvCount(recvCount), srcCount(srcCount)
        {}

        template <typename T>
        void operator()(oneapi::internal::Typelist<T>)
        {
            auto src = srcUnivers.get<T>().toSycl();
            auto dst = dstUnivers.get<T>().toSycl();
            ccl_request_t request;
            ccl_allgatherv(&src, srcCount, &dst, recvCount, get_ccl_datatype<T>(), NULL, NULL, stream, &request);
            ccl_wait(request);
        }
    };

public:
    static void allGatherV(cl::sycl::queue & queue, ccl_stream_t & stream, oneapi::internal::UniversalBuffer & dst, size_t * recvCount,
                           oneapi::internal::UniversalBuffer & src, size_t srcCount)
    {
        Execute op(queue, stream, dst, recvCount, src, srcCount);
        oneapi::internal::TypeDispatcher::dispatch(dst.type(), op);
    }
};

/** @} */
} // namespace interface1

using interface1::BufferAllReducer;
using interface1::BufferAllGatherer;

} // namespace internal
} // namespace comm
} // namespace preview
} // namespace daal

#endif
