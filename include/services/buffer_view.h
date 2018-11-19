/* file: buffer_view.h */
/*******************************************************************************
* Copyright 2014-2018 Intel Corporation.
*
* This software and the related documents are Intel copyrighted  materials,  and
* your use of  them is  governed by the  express license  under which  they were
* provided to you (License).  Unless the License provides otherwise, you may not
* use, modify, copy, publish, distribute,  disclose or transmit this software or
* the related documents without Intel's prior written permission.
*
* This software and the related documents  are provided as  is,  with no express
* or implied  warranties,  other  than those  that are  expressly stated  in the
* License.
*******************************************************************************/

/*
//++
//  Implementation of a dummy base class needed to fix ABI inconsistency between
//  Visual Studio* 2012 and 2013.
//--
*/

#ifndef __DAAL_SERVICES_BUFFER_VIEW_H__
#define __DAAL_SERVICES_BUFFER_VIEW_H__

#include "services/daal_defines.h"
#include "services/collection.h"

namespace daal
{
namespace services
{

/**
 * \brief Contains version 1.0 of Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) interface.
 */
namespace interface1
{

/**
 * @ingroup memory
 * @{
 */

/**
 *  <a name="DAAL-CLASS-SERVICES__BUFFERVIEW"></a>
 *  \brief   Class that refers to a contiguous sequence of objects, but
 *           doesn't control allocated memory buffer and objects lifetime,
 *           user is responsible for correct memory management and deallocation
 *  \tparam  T  Type of an object stored in the buffer
 */
template<typename T>
class BufferView : public Base
{
public:
    /**
     * Creates empty BufferView
     */
    BufferView() :
        _buffer(NULL),
        _size(0) { }

    /**
     * Creates BufferView from the raw data
     * \param[in]  buffer      The raw pointer to the buffer
     * \param[in]  bufferSize  The buffer size
     */
    explicit BufferView(T *buffer, size_t bufferSize) :
        _buffer(buffer),
        _size(bufferSize)
    {
        DAAL_ASSERT( _buffer );
    }

    /**
     *  Returns pointer to the underlying buffer serving as element storage
     *  \return Pointer to the array
     */
    T *data() const
    {
        return _buffer;
    }

    /**
     *  Size of a buffer
     *  \return Size of the buffer
     */
    size_t size() const
    {
        return _size;
    }

    /**
     *  Flag indicates that buffer is empty (its size is 0)
     *  \return Whether the buffer is empty
     */
    bool empty() const
    {
        return (_buffer == NULL) || (_size == 0);
    }

    /**
    *  Const element access
    *  \param[in] index Index of an accessed element
    *  \return    Const reference to the element
    */
    const T &operator [] (size_t index) const
    {
        DAAL_ASSERT( index < _size );
        return _buffer[index];
    }

    /**
     *  Element access
     *  \param[in] index Index of an accessed element
     *  \return    Reference to the element
     */
    T &operator [] (size_t index)
    {
        DAAL_ASSERT( index < _size );
        return _buffer[index];
    }

    /**
     * Gets the block of the current buffer
     * \param[in]  offset  The offset of the block
     * \param[in]  size    The size of the block
     * \return     New BufferView object for specified block
     */
    BufferView getBlock(size_t offset, size_t size) const
    {
        DAAL_ASSERT( offset + size <= _size );
        return BufferView<T>(_buffer + offset, size);
    }

private:
    T *_buffer;
    size_t _size;
};

/** @} */
} // namespace interface1

using interface1::BufferView;

} // namespace services
} // namespace daal

#endif
