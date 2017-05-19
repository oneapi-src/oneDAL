/* file: service_dnn_internal.h */
/*******************************************************************************
* Copyright 2014-2017 Intel Corporation
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
//  DNN service functions
//--
*/

#ifndef __SERVICE_DNN_INTERNAL_H__
#define __SERVICE_DNN_INTERNAL_H__

#include "service_defines.h"
#include "service_dnn.h"

#define ON_ERR(err)                                                                        \
{                                                                                          \
    if ((err) != E_SUCCESS)                                                                \
    {                                                                                      \
        if ((err) == E_MEMORY_ERROR)                                                       \
        { return services::Status(services::ErrorMemoryAllocationFailed);  }               \
        return  services::Status(services::ErrorConvolutionInternal);                      \
    }                                                                                      \
}

namespace daal
{
namespace internal
{

template<typename algorithmFPType, CpuType cpu>
class DnnLayout
{
public:
    typedef Dnn<algorithmFPType, cpu> dnn;

    DnnLayout() : layout(NULL), err(E_SUCCESS) {}

    DnnLayout(size_t dim, size_t *size, size_t *strides) : layout(NULL), err(E_SUCCESS)
    {
        err = dnn::xLayoutCreate(&layout, dim, size, strides);
    }

    DnnLayout(dnnPrimitive_t primitive, dnnResourceType_t resource) : layout(NULL), err(E_SUCCESS)
    {
        err = dnn::xLayoutCreateFromPrimitive(&layout, primitive, resource);
    }

    DnnLayout& operator=(DnnLayout&& source)
    {
        free();
        layout = source.layout;
        source.layout = NULL;
        return *this;
    }

    ~DnnLayout()
    {
        free();
    }

    void free()
    {
        if( layout != NULL )
        {
            dnn::xLayoutDelete(layout);
        }
    }

    dnnLayout_t& get() { return layout; }

    dnnError_t err;

protected:
    dnnLayout_t layout;
};

template<typename algorithmFPType, CpuType cpu>
class DnnBuffer
{
public:
    typedef Dnn<algorithmFPType, cpu> dnn;

    DnnBuffer(): buffer(0)
    {
    }

    DnnBuffer(dnnLayout_t layout)
    {
        err = dnn::xAllocateBuffer((void**)&buffer, layout);
    }

    ~DnnBuffer()
    {
        if (buffer)
        {
            dnn::xReleaseBuffer(buffer);
        }
    }

    algorithmFPType* get() { return buffer; }

    algorithmFPType* allocate(dnnLayout_t layout)
    {
        err = dnn::xAllocateBuffer((void**)&buffer, layout);
        return buffer;
    }

    dnnError_t err;
protected:
    algorithmFPType *buffer;
};

}
}

#endif
