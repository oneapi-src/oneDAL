/* file: compression_stream.cpp */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation.
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
//  Implementation of (de-)compression stream interface.
//--
*/

#include "compression_stream.h"

namespace daal
{
namespace data_management
{

typedef enum
{
    notallocated    = 0,
    externallocated = 1,
    internallocated = 2
} AllocationStateEnum;

typedef enum
{
    notprocessed = 0,
    compressed   = 1,
    decompressed = 2
} CompressionStateEnum;

class DAAL_EXPORT ReadWriteBlock : public DataBlock
{
public:
    explicit ReadWriteBlock() : DataBlock(), _w_off(0), _r_off(0), _allocState(notallocated)
    {}
    ReadWriteBlock(byte *ptr, size_t size) : DataBlock(ptr, size), _w_off(size), _r_off(0), _allocState(externallocated)
    {}
    explicit ReadWriteBlock(DataBlock *block) : DataBlock(block->getPtr(), block->getSize()), _w_off(block->getSize()), _r_off(0),
        _allocState(externallocated)
    {}
    explicit ReadWriteBlock(size_t size) : DataBlock(size), _w_off(0), _r_off(0), _allocState(internallocated)
    {
        byte *tmp_ptr = (byte *)daal::services::daal_malloc(size);
        setPtr(tmp_ptr);
    }

    virtual ~ReadWriteBlock()
    {
        if(_allocState != externallocated && _allocState != notallocated)
        {
            byte *tmp_ptr = getPtr();
            daal::services::daal_free(tmp_ptr);
        }
    }

    virtual AllocationStateEnum getAllocState() {return _allocState;}
    virtual size_t getWriteOffset() {return _w_off;}
    virtual size_t getReadOffset() {return _r_off;}

    virtual void setAllocState(AllocationStateEnum state) {_allocState = state;}
    virtual void setWriteOffset(size_t w_off) {_w_off = w_off;}
    virtual void setReadOffset(size_t r_off) {_r_off = r_off;}

private:
    AllocationStateEnum _allocState;
    size_t _w_off;
    size_t _r_off;
};

class DAAL_EXPORT CompressionBlock : public ReadWriteBlock
{
public:
    CompressionBlock() : ReadWriteBlock(), _comprState(notprocessed)
    {}
    CompressionBlock(byte *ptr, size_t size) : ReadWriteBlock(ptr, size), _comprState(notprocessed)
    {}
    explicit CompressionBlock(DataBlock *block) : ReadWriteBlock(block), _comprState(notprocessed)
    {}
    explicit CompressionBlock(size_t size) : ReadWriteBlock(size), _comprState(notprocessed)
    {}
    virtual ~CompressionBlock()
    {}

    virtual CompressionStateEnum getComprState() {return _comprState;}
    virtual void setComprState(CompressionStateEnum state) {_comprState = state;}

private:
    CompressionStateEnum _comprState;
};

typedef services::SharedPtr<CompressionBlock> CompressionBlockPtr;
typedef services::Collection<CompressionBlockPtr > CBC;

//compression stream realization
CompressionStream::CompressionStream(CompressorImpl *compr, size_t minSize) : _errors(new services::ErrorCollection()), _compressedDataSize(0), _writePos(0), _readPos(0), _blocks(NULL), _compressor(NULL), _minBlockSize(0)
{
    this->_errors->setCanThrow(false);
    if(compr == NULL)
    {
        this->_errors->add(services::ErrorIncorrectParameter);
        return;
    }
    if(minSize == 0)
    {
        this->_errors->add(services::ErrorIncorrectParameter);
        return;
    }
    _compressor = compr;
    _minBlockSize = minSize;
    _blocks = (void *) new CBC;
}

CompressionStream::~CompressionStream()
{
    if(_blocks) { delete (CBC *)_blocks; }
}

void CompressionStream::compressBlock(size_t pos)
{
    if(this->_errors->size() != 0)
    {
        return;
    }

    if(pos >= (*(CBC *)_blocks).size())
    {
        return;
    }

    if((*(CBC *)_blocks)[pos]->getComprState() == compressed)
    {
        return;
    }

    size_t tmpSize = (*(CBC *)_blocks)[pos]->getWriteOffset() > _minBlockSize ? _minBlockSize : (*
                                                                                                 (CBC *)_blocks)[pos]->getWriteOffset();

    _compressor->setInputDataBlock((*(CBC *)_blocks)[pos]->getPtr(), (*(CBC *)_blocks)[pos]->getWriteOffset(), 0);

    CBC tmpCollection;
    do
    {
        CompressionBlock *tmpBlock = new CompressionBlock(tmpSize);
        _compressor->run(tmpBlock->getPtr(), tmpBlock->getSize(), 0);
        tmpBlock->setWriteOffset(_compressor->getUsedOutputDataBlockSize());
        tmpBlock->setSize(_compressor->getUsedOutputDataBlockSize());
        tmpBlock->setComprState(compressed);
        tmpBlock->setAllocState(internallocated);
        tmpCollection.push_back(CompressionBlockPtr(tmpBlock));
    }
    while(_compressor->isOutputDataBlockFull());

    if(_compressor->getErrors()->size() != 0)
    {
        this->_errors->add(*(_compressor->getErrors()));
        tmpCollection.clear();
        return;
    }

    (*(CBC *)_blocks).erase(pos);
    (*(CBC *)_blocks).insert(pos, tmpCollection);
    _writePos = (*(CBC *)_blocks).size() - 1;
    tmpCollection.clear();
}

void CompressionStream::push_back(DataBlock *block)
{
    if(this->_errors->size() != 0)
    {
        return;
    }

    //checkParams;
    if ( block == NULL )
    {
        this->_errors->add(services::ErrorCompressionNullInputStream);
        return;
    }
    if ( block->getPtr() == NULL )
    {
        this->_errors->add(services::ErrorCompressionNullInputStream);
        return;
    }
    size_t inSize = block->getSize();
    if ( inSize == 0 )
    {
        this->_errors->add(services::ErrorCompressionEmptyInputStream);
        return;
    }
    //end checkParams;

    size_t colSize = (*(CBC *)_blocks).size();

    if(colSize > 0)
    {
        if(inSize <= (*(CBC *)_blocks)[_writePos]->getSize() - (*(CBC *)_blocks)[_writePos]->getWriteOffset())
        {
            byte *tmpPtr = (*(CBC *)_blocks)[_writePos]->getPtr();
            size_t tmpOffset = (*(CBC *)_blocks)[_writePos]->getWriteOffset();
            byte *blockPtr = block->getPtr();

            daal::services::daal_memcpy_s((void *)(tmpPtr + tmpOffset), inSize, (void *)blockPtr, inSize);

            (*(CBC *)_blocks)[_writePos]->setWriteOffset(tmpOffset + inSize);
            return;
        }
        else
        {
            compressBlock(_writePos);
        }
    }

    if(inSize >= _minBlockSize)
    {
        CompressionBlock *tmpBlock = new CompressionBlock(block);
        (*(CBC *)_blocks).push_back(CompressionBlockPtr(tmpBlock));
        _writePos = (*(CBC *)_blocks).size() - 1;
        compressBlock(_writePos);
    }
    else
    {
        CompressionBlock *tmpBlock = new CompressionBlock(_minBlockSize);
        byte *tmpPtr = tmpBlock->getPtr();
        byte *blockPtr = block->getPtr();

        daal::services::daal_memcpy_s((void *)tmpPtr, inSize, (void *)blockPtr, inSize);

        tmpBlock->setWriteOffset(inSize);
        (*(CBC *)_blocks).push_back(CompressionBlockPtr(tmpBlock));
        _writePos = (*(CBC *)_blocks).size() - 1;
    }
}

DataBlockCollectionPtr CompressionStream::getCompressedBlocksCollection()
{
    compressBlock(_writePos);

    DataBlockCollectionPtr retBlocks = DataBlockCollectionPtr(new DataBlockCollection);
    for(size_t i = 0; i < (*(CBC *)_blocks).size(); i++)
    {
        retBlocks->push_back(DataBlockPtr((*(CBC *)_blocks)[i]));
    }
    (*(CBC *)_blocks).clear();
    _writePos = 0;
    _readPos = 0;
    return retBlocks;
}

size_t CompressionStream::getCompressedDataSize()
{
    if(this->_errors->size() != 0)
    {
        return 0;
    }
    //    for(int i = 0; i < (*(CBC*)_blocks).size(); i++)
    //    {
    compressBlock(_writePos);
    //    }
    _compressedDataSize = 0;
    for(size_t i = 0; i < (*(CBC *)_blocks).size(); i++)
    {
        _compressedDataSize += (*(CBC *)_blocks)[i]->getWriteOffset() - (*(CBC *)_blocks)[i]->getReadOffset();
    }

    return _compressedDataSize;
}

size_t CompressionStream::copyCompressedArray(byte *ptr, size_t size)
{
    if(this->_errors->size() != 0)
    {
        return 0;
    }
    //checkParams;
    if ( ptr == NULL )
    {
        this->_errors->add(services::ErrorCompressionNullOutputStream);
        return 0;
    }
    if ( size == 0 )
    {
        this->_errors->add(services::ErrorCompressionEmptyOutputStream);
        return 0;
    }
    //end checkParams;


    size_t readSize = 0;
    size_t leftSize = size;
    byte *tmpPtr;

    if(_readPos == (*(CBC *)_blocks).size())
    {
        return readSize;
    }

    do
    {
        compressBlock(_readPos);
        size_t availSize = (*(CBC *)_blocks)[_readPos]->getWriteOffset() - (*(CBC *)_blocks)[_readPos]->getReadOffset();

        if(availSize == 0)
        {
            (*(CBC *)_blocks).erase(_readPos);
            continue;
        }

        tmpPtr = (*(CBC *)_blocks)[_readPos]->getPtr() + (*(CBC *)_blocks)[_readPos]->getReadOffset();

        size_t rs = leftSize > availSize ? availSize : leftSize;

        daal::services::daal_memcpy_s((void *)(ptr + readSize), rs, (void *)tmpPtr, rs);

        (*(CBC *)_blocks)[_readPos]->setReadOffset((*(CBC *)_blocks)[_readPos]->getReadOffset() + rs);

        availSize -= rs;

        if(!availSize)
        {
            (*(CBC *)_blocks).erase(_readPos);
        }
        readSize += rs;
        leftSize -= rs;

    }
    while(readSize < size && _readPos < (*(CBC *)_blocks).size());

    if((*(CBC *)_blocks).size())
    {
        _writePos = (*(CBC *)_blocks).size() - 1;
    }
    else
    {
        _writePos = 0;
    }
    return readSize;

}

//decompression stream realization
DecompressionStream::DecompressionStream(DecompressorImpl *compr,
                                         size_t minSize) : _errors(new services::ErrorCollection()), _decompressedDataSize(0), _writePos(0), _readPos(0), _blocks(NULL), _decompressor(NULL), _minBlockSize(0)
{
    this->_errors->setCanThrow(false);
    if(compr == NULL)
    {
        this->_errors->add(services::ErrorIncorrectParameter);
        return;
    }
    if(minSize == 0)
    {
        this->_errors->add(services::ErrorIncorrectParameter);
        return;
    }
    _decompressor = compr;
    _minBlockSize = minSize;
    _blocks = (void *) new CBC;
}

DecompressionStream::~DecompressionStream()
{
    if(_blocks) { delete (CBC *)_blocks; }
}

void DecompressionStream::decompressBlock(size_t pos)
{
    if(this->_errors->size() != 0)
    {
        return;
    }

    if((*(CBC *)_blocks)[pos]->getComprState() == decompressed)
    {
        return;
    }

    size_t tmpSize = (*(CBC *)_blocks)[pos]->getWriteOffset() > _minBlockSize ? _minBlockSize : (*
                                                                                                 (CBC *)_blocks)[pos]->getWriteOffset();
    _decompressor->setInputDataBlock((*(CBC *)_blocks)[pos]->getPtr(), (*(CBC *)_blocks)[pos]->getWriteOffset(), 0);

    CBC tmpCollection;
    do
    {
        CompressionBlock *tmpBlock = new CompressionBlock(tmpSize);
        _decompressor->run(tmpBlock->getPtr(), tmpBlock->getSize(), 0);
        tmpBlock->setWriteOffset(_decompressor->getUsedOutputDataBlockSize());
        tmpBlock->setSize(_decompressor->getUsedOutputDataBlockSize());
        tmpBlock->setComprState(decompressed);
        tmpBlock->setAllocState(internallocated);
        tmpCollection.push_back(CompressionBlockPtr(tmpBlock));
    }
    while(_decompressor->isOutputDataBlockFull());

    if(_decompressor->getErrors()->size() != 0)
    {
        this->_errors->add(*(_decompressor->getErrors()));
        tmpCollection.clear();
        return;
    }

    (*(CBC *)_blocks).erase(pos);
    (*(CBC *)_blocks).insert(pos, tmpCollection);
    _writePos = (*(CBC *)_blocks).size() - 1;
    tmpCollection.clear();
}

void DecompressionStream::push_back(DataBlock *block)
{
    if(this->_errors->size() != 0)
    {
        return;
    }
    //checkParams;
    if ( block == NULL )
    {
        this->_errors->add(services::ErrorCompressionNullInputStream);
        return;
    }
    if ( block->getPtr() == NULL )
    {
        this->_errors->add(services::ErrorCompressionNullInputStream);
        return;
    }
    size_t inSize = block->getSize();
    if ( inSize == 0 )
    {
        this->_errors->add(services::ErrorCompressionEmptyInputStream);
        return;
    }

    //end checkParams;
    CompressionBlock *tmpBlock = new CompressionBlock(block);
    (*(CBC *)_blocks).push_back(CompressionBlockPtr(tmpBlock));
    _writePos = (*(CBC *)_blocks).size() - 1;
    decompressBlock(_writePos);
}

DataBlockCollectionPtr DecompressionStream::getDecompressedBlocksCollection()
{
    getDecompressedDataSize();

    DataBlockCollectionPtr retBlocks = DataBlockCollectionPtr(new DataBlockCollection);
    for(size_t i = 0; i < (*(CBC *)_blocks).size(); i++)
    {
        retBlocks->push_back(DataBlockPtr((*(CBC *)_blocks)[i]));
    }
    (*(CBC *)_blocks).clear();
    _writePos = 0;
    _readPos = 0;
    return retBlocks;
}

size_t DecompressionStream::copyDecompressedArray(byte *ptr, size_t size)
{
    if(this->_errors->size() != 0)
    {
        return 0;
    }
    //checkParams;
    if ( ptr == NULL )
    {
        this->_errors->add(services::ErrorCompressionNullOutputStream);
        return 0;
    }
    if ( size == 0 )
    {
        this->_errors->add(services::ErrorCompressionEmptyOutputStream);
        return 0;
    }
    //end checkParams;

    size_t readSize = 0;
    size_t leftSize = size;
    byte *tmpPtr;

    if(_readPos == (*(CBC *)_blocks).size())
    {
        return readSize;
    }

    do
    {
        decompressBlock(_readPos);
        size_t availSize = (*(CBC *)_blocks)[_readPos]->getWriteOffset() - (*(CBC *)_blocks)[_readPos]->getReadOffset();

        if(availSize == 0)
        {
            (*(CBC *)_blocks).erase(_readPos);
            continue;
        }

        tmpPtr = (*(CBC *)_blocks)[_readPos]->getPtr() + (*(CBC *)_blocks)[_readPos]->getReadOffset();

        size_t rs = leftSize > availSize ? availSize : leftSize;

        daal::services::daal_memcpy_s((void *)(ptr + readSize), rs, (void *)tmpPtr, rs);

        (*(CBC *)_blocks)[_readPos]->setReadOffset((*(CBC *)_blocks)[_readPos]->getReadOffset() + rs);

        availSize -= rs;

        if(!availSize)
        {
            (*(CBC *)_blocks).erase(_readPos);
        }
        readSize += rs;
        leftSize -= rs;

    }
    while(readSize < size && _readPos < (*(CBC *)_blocks).size());

    return readSize;
}

size_t DecompressionStream::getDecompressedDataSize()
{
    if(this->_errors->size() != 0)
    {
        return 0;
    }

    for(int i = 0; i < (*(CBC *)_blocks).size(); i++)
    {
        decompressBlock(i);
    }
    _decompressedDataSize = 0;
    for(size_t i = 0; i < (*(CBC *)_blocks).size(); i++)
    {
        _decompressedDataSize += (*(CBC *)_blocks)[i]->getWriteOffset();
    }

    return _decompressedDataSize;
}

} //namespace data_management
} //namespace daal
