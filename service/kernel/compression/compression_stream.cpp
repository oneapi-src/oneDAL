/* file: compression_stream.cpp */
/*******************************************************************************
* Copyright 2014-2016 Intel Corporation
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

typedef services::Collection<services::SharedPtr<CompressionBlock> > CBC;

//compression stream realization
CompressionStream::CompressionStream(CompressorImpl *compr, size_t minSize) : _errors(new services::ErrorCollection())
{
    this->_errors->setCanThrow(false);
    blocks = NULL;
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
    compressor = compr;
    _compressedDataSize = 0;
    writePos = 0;
    readPos = 0;
    _minBlockSize = minSize;
    blocks = (void *) new CBC;
}

CompressionStream::~CompressionStream()
{
    if(blocks) { delete (CBC *)blocks; }
}

void CompressionStream::compressBlock(size_t pos)
{
    if(this->_errors->size() != 0)
    {
        return;
    }

    if(pos >= (*(CBC *)blocks).size())
    {
        return;
    }

    if((*(CBC *)blocks)[pos]->getComprState() == compressed)
    {
        return;
    }

    size_t tmpSize = (*(CBC *)blocks)[pos]->getWriteOffset() > _minBlockSize ? _minBlockSize : (*
                                                                                                (CBC *)blocks)[pos]->getWriteOffset();

    compressor->setInputDataBlock((*(CBC *)blocks)[pos]->getPtr(), (*(CBC *)blocks)[pos]->getWriteOffset(), 0);

    CBC tmpCollection;
    do
    {
        CompressionBlock *tmpBlock = new CompressionBlock(tmpSize);
        compressor->run(tmpBlock->getPtr(), tmpBlock->getSize(), 0);
        tmpBlock->setWriteOffset(compressor->getUsedOutputDataBlockSize());
        tmpBlock->setSize(compressor->getUsedOutputDataBlockSize());
        tmpBlock->setComprState(compressed);
        tmpBlock->setAllocState(internallocated);
        tmpCollection.push_back(services::SharedPtr<CompressionBlock>(tmpBlock));
    }
    while(compressor->isOutputDataBlockFull());

    if(compressor->getErrors()->size() != 0)
    {
        this->_errors->add(*(compressor->getErrors()));
        tmpCollection.clear();
        return;
    }

    (*(CBC *)blocks).erase(pos);
    (*(CBC *)blocks).insert(pos, tmpCollection);
    writePos = (*(CBC *)blocks).size() - 1;
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

    size_t colSize = (*(CBC *)blocks).size();

    if(colSize > 0)
    {
        if(inSize <= (*(CBC *)blocks)[writePos]->getSize() - (*(CBC *)blocks)[writePos]->getWriteOffset())
        {
            byte *tmpPtr = (*(CBC *)blocks)[writePos]->getPtr();
            size_t tmpOffset = (*(CBC *)blocks)[writePos]->getWriteOffset();
            byte *blockPtr = block->getPtr();

            daal::services::daal_memcpy_s((void *)(tmpPtr + tmpOffset), inSize, (void *)blockPtr, inSize);

            (*(CBC *)blocks)[writePos]->setWriteOffset(tmpOffset + inSize);
            return;
        }
        else
        {
            compressBlock(writePos);
        }
    }

    if(inSize >= _minBlockSize)
    {
        CompressionBlock *tmpBlock = new CompressionBlock(block);
        (*(CBC *)blocks).push_back(services::SharedPtr<CompressionBlock>(tmpBlock));
        writePos = (*(CBC *)blocks).size() - 1;
        compressBlock(writePos);
    }
    else
    {
        CompressionBlock *tmpBlock = new CompressionBlock(_minBlockSize);
        byte *tmpPtr = tmpBlock->getPtr();
        byte *blockPtr = block->getPtr();

        daal::services::daal_memcpy_s((void *)tmpPtr, inSize, (void *)blockPtr, inSize);

        tmpBlock->setWriteOffset(inSize);
        (*(CBC *)blocks).push_back(services::SharedPtr<CompressionBlock>(tmpBlock));
        writePos = (*(CBC *)blocks).size() - 1;
    }
}

services::SharedPtr<DataBlockCollection> CompressionStream::getCompressedBlocksCollection()
{
    compressBlock(writePos);

    services::SharedPtr<DataBlockCollection> retBlocks = services::SharedPtr<DataBlockCollection>(new DataBlockCollection);
    for(size_t i = 0; i < (*(CBC *)blocks).size(); i++)
    {
        retBlocks->push_back(services::SharedPtr<DataBlock>((*(CBC *)blocks)[i]));
    }
    (*(CBC *)blocks).clear();
    writePos = 0;
    readPos = 0;
    return retBlocks;
}

size_t CompressionStream::getCompressedDataSize()
{
    if(this->_errors->size() != 0)
    {
        return 0;
    }
    //    for(int i = 0; i < (*(CBC*)blocks).size(); i++)
    //    {
    compressBlock(writePos);
    //    }
    _compressedDataSize = 0;
    for(size_t i = 0; i < (*(CBC *)blocks).size(); i++)
    {
        _compressedDataSize += (*(CBC *)blocks)[i]->getWriteOffset() - (*(CBC *)blocks)[i]->getReadOffset();
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

    if(readPos == (*(CBC *)blocks).size())
    {
        return readSize;
    }

    do
    {
        compressBlock(readPos);
        size_t availSize = (*(CBC *)blocks)[readPos]->getWriteOffset() - (*(CBC *)blocks)[readPos]->getReadOffset();

        if(availSize == 0)
        {
            (*(CBC *)blocks).erase(readPos);
            continue;
        }

        tmpPtr = (*(CBC *)blocks)[readPos]->getPtr() + (*(CBC *)blocks)[readPos]->getReadOffset();

        size_t rs = leftSize > availSize ? availSize : leftSize;

        daal::services::daal_memcpy_s((void *)(ptr + readSize), rs, (void *)tmpPtr, rs);

        (*(CBC *)blocks)[readPos]->setReadOffset((*(CBC *)blocks)[readPos]->getReadOffset() + rs);

        availSize -= rs;

        if(!availSize)
        {
            (*(CBC *)blocks).erase(readPos);
        }
        readSize += rs;
        leftSize -= rs;

    }
    while(readSize < size && readPos < (*(CBC *)blocks).size());

    if((*(CBC *)blocks).size())
    {
        writePos = (*(CBC *)blocks).size() - 1;
    }
    else
    {
        writePos = 0;
    }
    return readSize;

}

//decompression stream realization
DecompressionStream::DecompressionStream(DecompressorImpl *compr,
                                         size_t minSize) : _errors(new services::ErrorCollection())
{
    this->_errors->setCanThrow(false);
    blocks = NULL;
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
    decompressor = compr;
    _decompressedDataSize = 0;
    writePos = 0;
    readPos = 0;
    _minBlockSize = minSize;
    blocks = (void *) new CBC;
}

DecompressionStream::~DecompressionStream()
{
    if(blocks) { delete (CBC *)blocks; }
}

void DecompressionStream::decompressBlock(size_t pos)
{
    if(this->_errors->size() != 0)
    {
        return;
    }

    if((*(CBC *)blocks)[pos]->getComprState() == decompressed)
    {
        return;
    }

    size_t tmpSize = (*(CBC *)blocks)[pos]->getWriteOffset() > _minBlockSize ? _minBlockSize : (*
                                                                                                (CBC *)blocks)[pos]->getWriteOffset();
    decompressor->setInputDataBlock((*(CBC *)blocks)[pos]->getPtr(), (*(CBC *)blocks)[pos]->getWriteOffset(), 0);

    CBC tmpCollection;
    do
    {
        CompressionBlock *tmpBlock = new CompressionBlock(tmpSize);
        decompressor->run(tmpBlock->getPtr(), tmpBlock->getSize(), 0);
        tmpBlock->setWriteOffset(decompressor->getUsedOutputDataBlockSize());
        tmpBlock->setSize(decompressor->getUsedOutputDataBlockSize());
        tmpBlock->setComprState(decompressed);
        tmpBlock->setAllocState(internallocated);
        tmpCollection.push_back(services::SharedPtr<CompressionBlock>(tmpBlock));
    }
    while(decompressor->isOutputDataBlockFull());

    if(decompressor->getErrors()->size() != 0)
    {
        this->_errors->add(*(decompressor->getErrors()));
        tmpCollection.clear();
        return;
    }

    (*(CBC *)blocks).erase(pos);
    (*(CBC *)blocks).insert(pos, tmpCollection);
    writePos = (*(CBC *)blocks).size() - 1;
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
    (*(CBC *)blocks).push_back(services::SharedPtr<CompressionBlock>(tmpBlock));
    writePos = (*(CBC *)blocks).size() - 1;
    decompressBlock(writePos);
}

services::SharedPtr<DataBlockCollection> DecompressionStream::getDecompressedBlocksCollection()
{
    getDecompressedDataSize();

    services::SharedPtr<DataBlockCollection> retBlocks = services::SharedPtr<DataBlockCollection>(new DataBlockCollection);
    for(size_t i = 0; i < (*(CBC *)blocks).size(); i++)
    {
        retBlocks->push_back(services::SharedPtr<DataBlock>((*(CBC *)blocks)[i]));
    }
    (*(CBC *)blocks).clear();
    writePos = 0;
    readPos = 0;
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

    if(readPos == (*(CBC *)blocks).size())
    {
        return readSize;
    }

    do
    {
        decompressBlock(readPos);
        size_t availSize = (*(CBC *)blocks)[readPos]->getWriteOffset() - (*(CBC *)blocks)[readPos]->getReadOffset();

        if(availSize == 0)
        {
            (*(CBC *)blocks).erase(readPos);
            continue;
        }

        tmpPtr = (*(CBC *)blocks)[readPos]->getPtr() + (*(CBC *)blocks)[readPos]->getReadOffset();

        size_t rs = leftSize > availSize ? availSize : leftSize;

        daal::services::daal_memcpy_s((void *)(ptr + readSize), rs, (void *)tmpPtr, rs);

        (*(CBC *)blocks)[readPos]->setReadOffset((*(CBC *)blocks)[readPos]->getReadOffset() + rs);

        availSize -= rs;

        if(!availSize)
        {
            (*(CBC *)blocks).erase(readPos);
        }
        readSize += rs;
        leftSize -= rs;

    }
    while(readSize < size && readPos < (*(CBC *)blocks).size());

    return readSize;
}

size_t DecompressionStream::getDecompressedDataSize()
{
    if(this->_errors->size() != 0)
    {
        return 0;
    }

    for(int i = 0; i < (*(CBC *)blocks).size(); i++)
    {
        decompressBlock(i);
    }
    _decompressedDataSize = 0;
    for(size_t i = 0; i < (*(CBC *)blocks).size(); i++)
    {
        _decompressedDataSize += (*(CBC *)blocks)[i]->getWriteOffset();
    }

    return _decompressedDataSize;
}

} //namespace data_management
} //namespace daal
