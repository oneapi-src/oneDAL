/* file: reducer.cpp */
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

#include "src/sycl/dpcpp_reducer.h"
#include "services/env_detect.h"
#include "src/externals/service_ittnotify.h"

#include <cfloat>

namespace daal
{
namespace oneapi
{
namespace internal
{
namespace math
{
DAAL_ITTNOTIFY_DOMAIN(daal.oneapi.internal.math.Reducer);

namespace impl
{
    template<Reducer::BinaryOp op, typename algorithmFPType> 
    struct BinaryOpFunctor 
    {
        constexpr algorithmFPType init_value = 0;
        constexpr algorithmFPType operator() (algorithmFPType a, algorithmFPType b) const
        {
            return a + b;
        }
    };

    template<typename algorithmFPType>
    struct BinaryOpFunctor<Reducer::BinaryOp::MIN, algorithmFPType>
    {
        constexpr algorithmFPType init_value = FLT_MAX; 
        constexpr algorithmFPType operator() (algorithmFPType a, algorithmFPType b) const
        {
            return (a < b) ? a : b;
        }
    };

    template<typename algorithmFPType>
    struct BinaryOpFunctor<Reducer::BinaryOp::MAX, algorithmFPType>
    {
        constexpr algorithmFPType init_value = - FLT_MAX; 
        constexpr algorithmFPType operator() (algorithmFPType a, algorithmFPType b) const
        {
            return (a < b) ? b : a;
        }
    };

    template<Reducer::BinaryOp op> 
    struct UnaryOpFunctor 
    {
        template<typename algorithmFPType>
        constexpr algorithmFPType operator() (algorithmFPType a) const
        {
            return a;
        }
    };

    template<> 
    struct UnaryOpFunctor<Reducer::BinaryOp::SUM_OF_SQUARES>
    {
        template<typename algorithmFPType>
        constexpr algorithmFPType operator() (algorithmFPType a) const
        {
            return a * a;
        }
    };

    template<Reducer::BinaryOp op, typename algorithmFPType, bool RowMajorLayout = true>
    struct ReducerSinglepassKernel
    {
        friend class Reducer;
        constexpr BinaryOpFunctor<op> binary;
        constexpr UnaryOpFunctor<op> unary;
        
        void operator() (cl::sycl::nd_item<2> idx) const
        {
            const std::uint32_t local_size = idx.get_local_size(0);

            std::uint32_t globalDim = 1;
            std::uint32_t localDim  = nVectors;

            if constexpr (RowMajorLayout)
            {
                globalDim = vectorSize;
                localDim  = 1;
            }

            std::uint32_t itemId  = idx.get_local_id(0);
            std::uint32_t groupId = idx.get_global_id(1);

            algorithmFPType el     = vectors[groupId * globalDim + itemId * localDim];
            partialReduces[itemId] = binary.init_value;

            for(std::uint32_t i = itemId; i < vectorSize; i += local_size)
            {
                el                     = vectors[groupId * globalDim + i * localDim];
                partialReduces[itemId] = binary(partialReduces[itemId], unary(el));
            }

            idx.barrier(cl::sycl::access::fence_space::local);

            for(std::uint32_t stride = local_size / 2; stride > 1; stride /= 2)
            {
                if(stride > itemId)
                {
                    partialReduces[itemId] = binary(partialReduces[itemId], partialReduces[itemId + stride]);
                }

                idx.barrier(cl::sycl::access::fence_space::local);
            }

            if(itemId == 0)
            {
                reduces[groupId] = binary(partialReduces[itemId], partialReduces[itemId + 1]);
            }
        }
        protected:
            bool vectorsAreRows;
            uint nVectors, vectorSize;
            cl::sycl::accessor<algorithmFPType, 1, cl::sycl::access::mode::read, cl::sycl::access::target::global_buffer> vectors;
            cl::sycl::accessor<algorithmFPType, 1, cl::sycl::access::mode::write, cl::sycl::access::target::global_buffer> reduces;
            cl::sycl::accessor<algorithmFPType, 1, cl::sycl::access::mode::read_write, cl::sycl::access::target::local> partialReduces;
    }

    template struct ReduceSinglepass<float>;
    template struct ReduceSinglePass<double>;
}

