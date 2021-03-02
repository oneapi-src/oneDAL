/* file: compression_stream.h */
/*******************************************************************************
* Copyright 2014-2021 Intel Corporation
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
/**
 * @ingroup data_compression
 * @{
 */
/**
 * \brief Collection of DataBlock-type elements.
 */
typedef services::Collection<services::SharedPtr<DataBlock> > DataBlockCollection;
typedef services::SharedPtr<DataBlockCollection> DataBlockCollectionPtr;

namespace interface1
{
/**
 * <a name="DAAL-CLASS-DATA_MANAGEMENT__COMPRESSIONSTREAM"></a>
 * \brief %CompressionStream class compresses input raw data by blocks.
 * <!-- \n<a href="DAAL-REF-COMPRESSION">Data compression usage model</a> -->
 *
 * \par References
 *      - \ref services::ErrorCompressionNullInputStream "Data compression error codes"
 *      - \ref CompressorImpl class
 */
class DAAL_EXPORT CompressionStream : public Base
{
public:
    /**
     * %CompressionStream constructor
     * \param compr Pointer to a specific Compressor used for compression
     * \param minSize Optional parameter, minimal size of internal data blocks
     */
    CompressionStream(CompressorImpl * compr, size_t minSize = 1024 * 64);
    virtual ~CompressionStream() DAAL_C11_OVERRIDE;

    /**
     * Writes the next DataBlock to %CompressionStream and compresses it
     * \param[in] inBlock  Pointer to the next DataBlock to be compressed
     */
    virtual void push_back(DataBlock * inBlock);

    /**
     * Writes the next DataBlock to %CompressionStream and compresses it
     * \param[in] inBlock  Pointer to the next DataBlock to be compressed
     */
    virtual void operator<<(DataBlock * inBlock) { push_back(inBlock); }
    /**
     * Writes the next DataBlock to %CompressionStream and compresses it
     * \param[in] inBlock  Next DataBlock to be compressed
     */
    virtual void operator<<(DataBlock inBlock) { push_back(&inBlock); }
    /**
     * Provides access to compressed data blocks stored in %CompressionStream
     * \return Pointer to an internal \ref DataBlockCollection
     */
    virtual DataBlockCollectionPtr getCompressedBlocksCollection();
    /**
     * Returns the size of compressed data stored in %CompressionStream
     * \return Size in bytes
     */
    virtual size_t getCompressedDataSize();
    /**
     * Copies compressed data stored in %CompressionStream to an external array
     * \param[out] outPtr Pointer to the array where compressed data is stored
     * \param[in] outSize Number of bytes available in external memory
     * \return Size of copied data in bytes
     */
    virtual size_t copyCompressedArray(byte * outPtr, size_t outSize);
    /**
     * Copies compressed data stored in %CompressionStream to an external DataBlock
     * \param[out] outBlock Reference to the DataBlock where compressed data is stored
     * \return Size of copied data in bytes
     */
    virtual size_t copyCompressedArray(DataBlock & outBlock) { return copyCompressedArray(outBlock.getPtr(), outBlock.getSize()); }

    services::SharedPtr<services::ErrorCollection> getErrors() { return _errors; }

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

/**
 * <a name="DAAL-CLASS-DATA_MANAGEMENT__DECOMPRESSIONSTREAM"></a>
 * \brief %DecompressionStream class decompresses compressed input data by blocks.
 * <!-- \n<a href="DAAL-REF-COMPRESSION">Data compression usage model</a> -->
 *
 * \par References
 *      - \ref services::ErrorCompressionNullInputStream "Data compression error codes"
 *      - \ref DecompressorImpl class
 */
class DAAL_EXPORT DecompressionStream : public Base
{
public:
    /**
     * \brief %DecompressionStream constructor
     * \param decompr Pointer to a specific Decompressor used for decompression
     * \param minSize Optional parameter, minimal size of internal data blocks
     */
    DecompressionStream(DecompressorImpl * decompr, size_t minSize = 1024 * 64);
    virtual ~DecompressionStream() DAAL_C11_OVERRIDE;
    /**
     * Writes the next compressed DataBlock to %DecompressionStream and decompresses it
     * \param[in] inBlock  Pointer to the next DataBlock to be decompressed
     */
    virtual void push_back(DataBlock * inBlock);
    /**
     * Writes the next compressed DataBlock to %DecompressionStream and decompresses it
     * \param[in] inBlock  Pointer to the next DataBlock to be decompressed
     */
    virtual void operator<<(DataBlock * inBlock) { push_back(inBlock); }
    /**
     * Writes the next compressed DataBlock to %DecompressionStream and decompresses it
     * \param[in] inBlock  Next DataBlock to be decompressed
     */
    virtual void operator<<(DataBlock inBlock) { push_back(&inBlock); }
    /**
     * Provides access to decompressed data blocks stored in %DecompressionStream
     * \return Pointer to internal \ref DataBlockCollection
     */
    virtual DataBlockCollectionPtr getDecompressedBlocksCollection();
    /**
     * Returns the size of decompressed data stored in %DecompressionStream
     * \return Size in bytes
     */
    virtual size_t getDecompressedDataSize();
    /**
     * Copies decompressed data stored in %DecompressionStream to an external array
     * \param[out] outPtr Pointer to the array where decompressed data is stored
     * \param[in] outSize Number of bytes available in external memory
     * \return Size of copied data in bytes
     */
    virtual size_t copyDecompressedArray(byte * outPtr, size_t outSize);
    /**
     * Copies decompressed data stored in %DecompressionStream to an external DataBlock
     * \param[out] outBlock Reference to the DataBlock where decompressed data is stored.
     *                      Size of DataBlock must be at least getDecompressedSize() bytes
     */
    virtual size_t copyDecompressedArray(DataBlock & outBlock) { return copyDecompressedArray(outBlock.getPtr(), outBlock.getSize()); }

    services::SharedPtr<services::ErrorCollection> getErrors() { return _errors; }

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
/** @} */

} //namespace data_management
} //namespace daal

#endif // __COMPRESSION_STREAM_H
