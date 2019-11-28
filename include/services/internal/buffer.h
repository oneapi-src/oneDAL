/* file: buffer.h */
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

#ifndef __SERVICES_INTERNAL_BUFFER_H__
#define __SERVICES_INTERNAL_BUFFER_H__

#include "services/base.h"
#include "services/buffer_view.h"

namespace daal
{
namespace services
{
namespace internal
{
/**
 * @ingroup memory
 * @{
 */

/**
 * <a name="DAAL-CLASS-SERVICES__INTERNAL__BUFFER"></a>
 * \brief  Class that provides simple memory management routines for handling blocks
 *         of continues memory, also provides automatic memory deallocation. Note this
 *         class doesn't provide functionality for objects constructions and simply allocates
 *         and deallocates memory. In case of objects consider Collection or ObjectPtrCollection
 * \tparam T Type of elements which are stored in the buffer
 */
template <typename T>
class Buffer : public Base
{
public:
    Buffer() : _buffer(NULL), _size(0) {}

    explicit Buffer(size_t size, services::Status * status = NULL)
    {
        services::Status localStatus = reallocate(size);
        services::internal::tryAssignStatusAndThrow(status, localStatus);
    }

    virtual ~Buffer() { destroy(); }

    void destroy()
    {
        services::daal_free((void *)_buffer);
        _buffer = NULL;
        _size   = 0;
    }

    services::Status reallocate(size_t size, bool copy = false)
    {
        if (_size == size)
        {
            return services::Status();
        }

        T * buffer = (T *)services::daal_calloc(sizeof(T) * size);
        if (!buffer)
        {
            return services::throwIfPossible(services::ErrorMemoryAllocationFailed);
        }

        if (copy)
        {
            for (size_t i = 0; i < _size; i++)
            {
                _buffer[i] = buffer[i];
            }
        }

        destroy();

        _size   = size;
        _buffer = buffer;
        return services::Status();
    }

    services::Status enlarge(size_t factor = 2, bool copy = false) { return reallocate(_size * factor, copy); }

    size_t size() const { return _size; }

    T * data() const { return _buffer; }

    T * offset(size_t elementsOffset) const
    {
        DAAL_ASSERT(elementsOffset <= _size);
        return _buffer + elementsOffset;
    }

    T & operator[](size_t index)
    {
        DAAL_ASSERT(index < _size);
        return _buffer[index];
    }

    const T & operator[](size_t index) const
    {
        DAAL_ASSERT(index < _size);
        return _buffer[index];
    }

    services::BufferView<T> view() const { return services::BufferView<T>(_buffer, _size); }

private:
    Buffer(const Buffer &);
    Buffer & operator=(const Buffer &);

private:
    T * _buffer;
    size_t _size;
};
/** @} */

} // namespace internal
} // namespace services
} // namespace daal

#endif
