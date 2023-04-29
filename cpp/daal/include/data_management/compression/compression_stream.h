/* file: compression_stream.h */
/*******************************************************************************
* Copyright 2014 Intel Corporation
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
//  Implementation of the compression and decompression stream interface.
//--
*/

#ifndef __COMPRESSION_STREAM_H__
#define __COMPRESSION_STREAM_H__

#include "services/base.h"
#include "data_management/compression/compression.h"
#include "data_management/data/data_block.h"
#include "services/collection.h"

namespace daal
{
namespace data_management
{
typedef services::Collection<services::SharedPtr<DataBlock> > DataBlockCollection;
typedef services::SharedPtr<DataBlockCollection> DataBlockCollectionPtr;

namespace interface1
{
class DAAL_EXPORT CompressionStream : public Base
{
public:
    DAAL_DEPRECATED CompressionStream(CompressorImpl * compr, size_t minSize = 1024 * 64);
    DAAL_DEPRECATED_VIRTUAL virtual ~CompressionStream() DAAL_C11_OVERRIDE;

    DAAL_DEPRECATED_VIRTUAL virtual void push_back(DataBlock * inBlock);

    DAAL_DEPRECATED_VIRTUAL virtual void operator<<(DataBlock * inBlock) { push_back(inBlock); }

    DAAL_DEPRECATED_VIRTUAL virtual void operator<<(DataBlock inBlock) { push_back(&inBlock); }

    DAAL_DEPRECATED_VIRTUAL virtual DataBlockCollectionPtr getCompressedBlocksCollection();

    DAAL_DEPRECATED_VIRTUAL virtual size_t getCompressedDataSize();

    DAAL_DEPRECATED_VIRTUAL virtual size_t copyCompressedArray(byte * outPtr, size_t outSize);

    DAAL_DEPRECATED_VIRTUAL virtual size_t copyCompressedArray(DataBlock & outBlock)
    {
        return copyCompressedArray(outBlock.getPtr(), outBlock.getSize());
    }

    DAAL_DEPRECATED services::SharedPtr<services::ErrorCollection> getErrors() { return _errors; }

private:
    void * _blocks;

    CompressorImpl * _compressor;
    size_t _compressedDataSize;
    size_t _minBlockSize;

    size_t _writePos;
    size_t _readPos;

    void compressBlock(size_t pos);

    services::SharedPtr<services::ErrorCollection> _errors;
};

class DAAL_EXPORT DecompressionStream : public Base
{
public:
    DAAL_DEPRECATED DecompressionStream(DecompressorImpl * decompr, size_t minSize = 1024 * 64);
    DAAL_DEPRECATED_VIRTUAL virtual ~DecompressionStream() DAAL_C11_OVERRIDE;

    DAAL_DEPRECATED_VIRTUAL virtual void push_back(DataBlock * inBlock);

    DAAL_DEPRECATED_VIRTUAL virtual void operator<<(DataBlock * inBlock) { push_back(inBlock); }

    DAAL_DEPRECATED_VIRTUAL virtual void operator<<(DataBlock inBlock) { push_back(&inBlock); }

    DAAL_DEPRECATED_VIRTUAL virtual DataBlockCollectionPtr getDecompressedBlocksCollection();

    DAAL_DEPRECATED_VIRTUAL virtual size_t getDecompressedDataSize();

    DAAL_DEPRECATED_VIRTUAL virtual size_t copyDecompressedArray(byte * outPtr, size_t outSize);

    DAAL_DEPRECATED_VIRTUAL virtual size_t copyDecompressedArray(DataBlock & outBlock)
    {
        return copyDecompressedArray(outBlock.getPtr(), outBlock.getSize());
    }

    DAAL_DEPRECATED services::SharedPtr<services::ErrorCollection> getErrors() { return _errors; }

private:
    void * _blocks;

    DecompressorImpl * _decompressor;
    size_t _decompressedDataSize;
    size_t _minBlockSize;

    size_t _writePos;
    size_t _readPos;

    void decompressBlock(size_t pos);

    services::SharedPtr<services::ErrorCollection> _errors;
};
} // namespace interface1
using interface1::CompressionStream;
using interface1::DecompressionStream;

} //namespace data_management
} //namespace daal

#endif // __COMPRESSION_STREAM_H
