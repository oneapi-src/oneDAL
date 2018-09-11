/* file: memory_block.h */
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

#ifndef __MEMORY_BLOCK_H__
#define __MEMORY_BLOCK_H__

#include "services/daal_defines.h"
#include "data_management/data/data_serialize.h"
#include "data_management/data/data_archive.h"
#include "services/daal_shared_ptr.h"

namespace daal
{
namespace data_management
{

namespace interface1
{

/**
*  <a name="DAAL-CLASS-DATA_MANAGEMENT__MEMORYBLOCK"></a>
*  \brief Serializable memory block, owner of the memory
*/
class DAAL_EXPORT MemoryBlock : public SerializationIface
{
public:
    DECLARE_SERIALIZABLE_TAG();

    DAAL_CAST_OPERATOR(MemoryBlock);

    /** Default constructor */
    MemoryBlock() : _size(0), _value(NULL)
    {
    }

    /** Constructs Memory Block object by allocating memory of size equal to the requested number of bytes
    * \param[in] n Number of bytes to allocate
    */
    MemoryBlock(size_t n);

    virtual ~MemoryBlock();

    /** Allocates given number of bytes.
    * Owned memory is reallocated if its size is less than required
    * \param[in] n Number of bytes to allocate
    * \return Reference to SharedPtr of the SerializationIface type
    */
    void reserve(size_t n);

    /**
    *  Returns pointer to the owned memory
    *  \return Pointer to the owned memory
    */
    byte* get()
    {
        return _value;
    }

    /**
    *  Returns pointer to the owned memory
    *  \return Pointer to the owned memory
    */
    const byte* get() const
    {
        return _value;
    }

    /**
    *  Returns the size of stored memory in bytes
    *  \return Number of stored bytes
    */
    size_t size() const
    {
        return _size;
    }

    /**
    *  Releases owned memory
    */
    void release();

protected:
    virtual services::Status serializeImpl(interface1::InputDataArchive  *arch) DAAL_C11_OVERRIDE
    {
        arch->set(_size);
        if(_size)
            arch->set(_value, _size);

        return services::Status();
    }

    virtual services::Status deserializeImpl(const interface1::OutputDataArchive *arch) DAAL_C11_OVERRIDE
    {
        size_t sz = 0;
        arch->set(sz);
        reserve(sz);
        if(sz)
            arch->set(_value, sz);

        return services::Status();
    }

protected:
    size_t _size;
    byte* _value;
};
typedef services::SharedPtr<MemoryBlock> MemoryBlockPtr;

} // namespace interface1
using interface1::MemoryBlock;
using interface1::MemoryBlockPtr;

}
}

#endif
