/* file: zlibcompression.h */
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
//  Implementation of the ZLIB DEFLATE compression and decompression interface.
//--
*/

#ifndef __ZLIBCOMPRESSION_H__
#define __ZLIBCOMPRESSION_H__
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
 * <a name="DAAL-CLASS-DATA_MANAGEMENT__ZLIBCOMPRESSIONPARAMETER"></a>
 *
 * \brief Parameter for zlib compression and decompression
 *
 * \snippet compression/zlibcompression.h ZlibCompressionParameter source code
 *
 * \par Enumerations
 *      - \ref CompressionLevel - %Compression level
 */
/* [ZlibCompressionParameter source code] */
class DAAL_EXPORT ZlibCompressionParameter : public data_management::CompressionParameter
{
public:
    /**
     * %ZlibCompressionParameter constructor
     * \param clevel    Optional parameter, CompressionLevel.
     *                  defaultLevel is equal to zlib compression level 6
     * \param bgzHeader Optional flag. True if a simple GZIP header is included, false otherwise
     */
    ZlibCompressionParameter(CompressionLevel clevel = defaultLevel, bool bgzHeader = false)
        : data_management::CompressionParameter(clevel), gzHeader(bgzHeader)
    {}

    ~ZlibCompressionParameter() {}

    bool gzHeader; /*!< Optional flag. True if simple GZIP header is included, false otherwise */
};
/* [ZlibCompressionParameter source code] */

/**
 * <a name="DAAL-CLASS-COMPRESSOR_ZLIB"></a>
 *
 * \brief Implementation of the Compressor class for the zlib compression method
 * <!-- \n<a href="DAAL-REF-COMPRESSION">Data compression usage model</a> -->
 *
 * \par References
 *      - \ref services::ErrorCompressionNullInputStream "Data compression error codes"
 *      - \ref ZlibCompressionParameter class
 */
template <>
class DAAL_EXPORT Compressor<zlib> : public data_management::CompressorImpl
{
public:
    /**
     * \brief Compressor<zlib> constructor
     */
    Compressor();
    ~Compressor();
    /**
     * Associates an input data block with a compressor
     * \param[in] inBlock Pointer to the data block to compress. Must be at least size+offset bytes
     * \param[in] size     Number of bytes to compress in inBlock
     * \param[in] offset   Offset in bytes, the starting position for compression in inBlock
     */
    void setInputDataBlock(byte * inBlock, size_t size, size_t offset);
    /**
     * Associates an input data block with a compressor
     * \param[in] inBlock Reference to the data block to compress
     */
    void setInputDataBlock(DataBlock & inBlock) { setInputDataBlock(inBlock.getPtr(), inBlock.getSize(), 0); }
    /**
     * Performs zlib compression of a data block
     * \param[out] outBlock Pointer to the data block where compression results are stored. Must be at least size+offset bytes
     * \param[in] size       Number of bytes available in outBlock
     * \param[in] offset     Offset in bytes, the starting position for compression in outBlock
     */
    void run(byte * outBlock, size_t size, size_t offset);
    /**
     * Performs zlib compression of a data block
     * \param[out] outBlock Reference to the data block where compression results are stored
     */
    void run(DataBlock & outBlock) { run(outBlock.getPtr(), outBlock.getSize(), 0); }

    ZlibCompressionParameter parameter; /*!< Zlib compression parameters structure */

protected:
    void initialize();

private:
    void * _strmp;
    int _flush;

    void finalizeCompression();
    void resetCompression();
};

/**
 * <a name="DAAL-CLASS-DECOMPRESSOR_ZLIB"></a>
 *
 * \brief Implementation of the Decompressor class for the zlib compression method
 * <!-- \n<a href="DAAL-REF-COMPRESSION">Data compression usage model</a> -->
 *
 * \par References
 *      - \ref services::ErrorCompressionNullInputStream "Data compression error codes"
 *      - \ref ZlibCompressionParameter class
 */
template <>
class DAAL_EXPORT Decompressor<zlib> : public data_management::DecompressorImpl
{
public:
    /**
     * \brief Decompressor<zlib> constructor
     */
    Decompressor();
    ~Decompressor();
    /**
     * Associates an input data block with a decompressor
     * \param[in] inBlock Pointer to the data block to decompress. Must be at least size+offset bytes
     * \param[in] size     Number of bytes to decompress in inBlock
     * \param[in] offset   Offset in bytes, the starting position for decompression in inBlock
     */
    void setInputDataBlock(byte * inBlock, size_t size, size_t offset);
    /**
     * Associates an input data block with a decompressor
     * \param[in] inBlock Reference to the data block to decompress
     */
    void setInputDataBlock(DataBlock & inBlock) { setInputDataBlock(inBlock.getPtr(), inBlock.getSize(), 0); }
    /**
     * Performs zlib decompression of a data block
     * \param[out] outBlock Pointer to the data block where decompression results are stored. Must be at least size+offset bytes
     * \param[in] size       Number of bytes available in outBlock
     * \param[in] offset     Offset in bytes, the starting position for decompression in outBlock
     */
    void run(byte * outBlock, size_t size, size_t offset);
    /**
     * Performs zlib decompression of a data block
     * \param[out] outBlock Reference to the data block where decompression results are stored
     */
    void run(DataBlock & outBlock) { run(outBlock.getPtr(), outBlock.getSize(), 0); }

    ZlibCompressionParameter parameter; /*!< Zlib compression parameters structure */

protected:
    void initialize();

private:
    void * _strmp;
    int _flush;

    void finalizeCompression();
    void resetCompression();
};
/** @} */
} // namespace interface1
using interface1::ZlibCompressionParameter;
using interface1::Compressor;
using interface1::Decompressor;

} //namespace data_management
} //namespace daal
#endif //__ZLIBCOMPRESSION_H
