/* file: data_block.h */
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
//  Implementation of the DataBlock type
//--
*/

#ifndef __DAAL_DATABLOCK_H__
#define __DAAL_DATABLOCK_H__

#include "services/base.h"
#include "services/daal_shared_ptr.h"

namespace daal
{
namespace data_management
{

namespace interface1
{
/**
 * @ingroup serialization
 * @{
 */
/**
 * <a name="DAAL-CLASS-DATA_MANAGEMENT__DATABLOCKIFACE"></a>
 * \brief Abstract interface class for a data management component responsible for a pointer to a byte array and its size.
 * This class declares the most general methods for data access.
 */
class DataBlockIface : public Base
{
public:
    virtual ~DataBlockIface() {}
    /**
    * Returns a pointer to a byte array stored in DataBlock
    * \return Pointer to the byte array stored in DataBlock
    */
    virtual byte *getPtr() const = 0;
    /**
    * Returns a pointer to a byte array stored in DataBlock
    * \return Pointer to the byte array stored in DataBlock
    */
    virtual services::SharedPtr<byte> getSharedPtr() const = 0;
    /**
     * Returns the size of a byte array stored in DataBlock
     * \return Size of the byte array stored in DataBlock
     */
    virtual size_t getSize() const = 0;
    /**
     * Sets a pointer to a byte array
     * \param[in] ptr Pointer to the byte array
     */
    virtual void setPtr(byte *ptr) = 0;
    /**
     * Sets a pointer to a byte array
     * \param[in] ptr Pointer to the byte array
     */
    virtual void setPtr(const services::SharedPtr<byte> &ptr) = 0;
    /**
     * Sets the size of a byte array
     * \param[in] size Size of the byte array
     */
    virtual void setSize(size_t size) = 0;
};
/**
 * <a name="DAAL-CLASS-DATA_MANAGEMENT__DATABLOCK"></a>
 * \brief Class that stores a pointer to a byte array and its size. Not responsible for memory management
 */
class DAAL_EXPORT DataBlock : public DataBlockIface
{
public:
    /**
     * Default constructor. Creates an empty DataBlock of zero size with a zero pointer to a byte array
     */
    DataBlock() : _ptr(), _size(0)
    {}
    /**
     * Constructor. Creates DataBlock with a user-defined byte array
     * \param ptr Pointer to the byte array
     * \param size Size of the byte array
     */
    DataBlock(byte * ptr, size_t size) : _ptr(ptr, services::EmptyDeleter()), _size(size)
    {}
    /**
     * Constructor. Creates DataBlock with a user-defined byte array
     * \param ptr Pointer to the byte array
     * \param size Size of the byte array
     */
    DataBlock(const services::SharedPtr<byte> &ptr, size_t size) : _ptr(ptr), _size(size)
    {}
    /**
     * Constructor. Creates an empty DataBlock of a predefined size
     * \param size Size of the byte array
     */
    DataBlock(size_t size) : _ptr(), _size(size)
    {}
    /**
     * Copy constructor. Copies a pointer and the size stored in another DataBlock
     * \param block Reference to DataBlock
     */
    DataBlock(const DataBlock &block)
    {
       _ptr = block._ptr;
       _size = block._size;
    }

    virtual ~DataBlock() {}

    virtual byte *getPtr() const DAAL_C11_OVERRIDE
    {
        return _ptr.get();
    }

    virtual services::SharedPtr<byte> getSharedPtr() const DAAL_C11_OVERRIDE
    {
        return _ptr;
    }

    virtual size_t getSize() const DAAL_C11_OVERRIDE
    {
        return _size;
    }

    virtual void setPtr(byte *ptr) DAAL_C11_OVERRIDE
    {
        _ptr = services::SharedPtr<byte>(ptr, services::EmptyDeleter());
    }

    virtual void setPtr(const services::SharedPtr<byte> &ptr) DAAL_C11_OVERRIDE
    {
        _ptr = ptr;
    }

    virtual void setSize(size_t size) DAAL_C11_OVERRIDE
    {
        _size = size;
    }

private:
    services::SharedPtr<byte> _ptr;
    size_t _size;
};
typedef services::SharedPtr<DataBlock> DataBlockPtr;
/** @} */
} // namespace interface1
using interface1::DataBlock;
using interface1::DataBlockPtr;
using interface1::DataBlockIface;
}
}

#endif
