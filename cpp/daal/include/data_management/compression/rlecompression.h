/* file: rlecompression.h */
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
//  Implementation of the run-length encoding interface.
//--
*/

#ifndef __RLECOMPRESSION_H__
#define __RLECOMPRESSION_H__
#include "data_management/compression/compression.h"

namespace daal
{
namespace data_management
{
namespace interface1
{
/**
 * @ingroup data_compression
 * @{
 */
/**
 * <a name="DAAL-CLASS-RLECOMPRESSIONPARAMETER"></a>
 *
 * \brief Parameter for run-length encoding and decoding.
 * A RLE encoded block may contain a header that consists of two sections: 1) decoded data size (4 bytes) and 2) encoded data size (4 bytes)
 *
 * \snippet compression/rlecompression.h RleCompressionParameter source code
 *
 */
/* [RleCompressionParameter source code] */
class DAAL_EXPORT RleCompressionParameter : public data_management::CompressionParameter
{
public:
    /**
     * RleCompressionParameter constructor
     * \param _isBlockHeader RLE block header presence flag. True if a RLE block header is present, false otherwise
     */
    RleCompressionParameter(bool _isBlockHeader = 1) : data_management::CompressionParameter(defaultLevel), isBlockHeader(_isBlockHeader) {}

    ~RleCompressionParameter() {}

    bool isBlockHeader; /*!< RLE block header presence flag. True if a RLE block header is present, false otherwise */
};
/* [RleCompressionParameter source code] */

/**
 * <a name="DAAL-CLASS-COMPRESSOR_RLE"></a>
 *
 * \brief Implementation of the Compressor class for the run-length encoding method
 * <!-- \n<a href="DAAL-REF-COMPRESSION">Data compression usage model</a> -->
 *
 * \par References
 *      - \ref services::ErrorCompressionNullInputStream "Data compression error codes"
 *      - \ref RleCompressionParameter class
 */
template <>
class DAAL_EXPORT Compressor<rle> : public data_management::CompressorImpl
{
public:
    /**
     * \brief Compressor<rle> constructor
     */
    Compressor();
    ~Compressor();
    /**
     * Associates an input data block with a compressor
     * \param[in] inBlock Pointer to the data block to encode. Must be at least size+offset bytes
     * \param[in] size     Number of bytes to encode in inBlock
     * \param[in] offset   Offset in bytes, the starting position for encoding in inBlock
     */
    void setInputDataBlock(byte * inBlock, size_t size, size_t offset);
    /**
     * Associates an input data block with a compressor
     * \param[in] inBlock Reference to the data block to encode
     */
    void setInputDataBlock(DataBlock & inBlock) { setInputDataBlock(inBlock.getPtr(), inBlock.getSize(), 0); }

    /**
     * Performs run-length encoding of a data block
     * \param[out] outBlock Pointer to the data block where encoding results are stored. Must be at least size+offset bytes
     * \param[in] size       Number of bytes available in outBlock
     * \param[in] offset     Offset in bytes, the starting position for encoding in outBlock
     */
    void run(byte * outBlock, size_t size, size_t offset);
    /**
     * Performs run-length encoding of a data block
     * \param[out] outBlock Reference to the data block where encoding results are stored
     */
    void run(DataBlock & outBlock) { run(outBlock.getPtr(), outBlock.getSize(), 0); }

    RleCompressionParameter parameter; /*!< RLE compression parameters structure */

protected:
    void initialize();

private:
    void * _next_in;
    size_t _avail_in;
    void * _next_out;
    size_t _avail_out;
    size_t _headBytes;

    void finalizeCompression();
};

/**
 * <a name="DAAL-CLASS-DECOMPRESSOR_RLE"></a>
 *
 * \brief Implementation of the Decompressor class for the run-length decoding method
 * <!-- \n<a href="DAAL-REF-COMPRESSION">Data compression usage model</a> -->
 *
 * \par References
 *      - \ref services::ErrorCompressionNullInputStream "Data compression error codes"
 *      - \ref RleCompressionParameter class
 */
template <>
class DAAL_EXPORT Decompressor<rle> : public data_management::DecompressorImpl
{
public:
    /**
     * \brief Decompressor<rle> constructor
     */
    Decompressor();
    ~Decompressor();
    /**
     * Associates an input data block with a decompressor
     * \param[in] inBlock Pointer to the data block to decode. Must be at least size+offset bytes
     * \param[in] size     Number of bytes to decode in inBlock
     * \param[in] offset   Offset in bytes, the starting position for decoding in inBlock
     */
    void setInputDataBlock(byte * inBlock, size_t size, size_t offset);
    /**
     * Associates an input data block with a decompressor
     * \param[in] inBlock Reference to the data block to decode
     */
    void setInputDataBlock(DataBlock & inBlock) { setInputDataBlock(inBlock.getPtr(), inBlock.getSize(), 0); }
    /**
     * Performs run-length decoding of a data block
     * \param[out] outBlock Pointer to the data block where decoding results are stored. Must be at least size+offset bytes
     * \param[in] size       Number of bytes available in outBlock
     * \param[in] offset     Offset in bytes, the starting position for decoding in outBlock
     */
    void run(byte * outBlock, size_t size, size_t offset);
    /**
     * Performs run-length decoding of a data block
     * \param[out] outBlock Reference to the data block where decoding results are stored
     */
    void run(DataBlock & outBlock) { run(outBlock.getPtr(), outBlock.getSize(), 0); }

    RleCompressionParameter parameter; /*!< RLE compression parameters structure */

protected:
    void initialize();

private:
    void * _next_in;
    size_t _avail_in;
    void * _next_out;
    size_t _avail_out;
    size_t _headBytes;

    void * _internalBuff;
    size_t _internalBuffOff;
    size_t _internalBuffLen;

    void finalizeCompression();
};
/** @} */
} // namespace interface1
using interface1::RleCompressionParameter;
using interface1::Compressor;
using interface1::Decompressor;

} //namespace data_management
} //namespace daal
#endif //__RLECOMPRESSION_H
