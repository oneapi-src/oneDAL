/* file: compression.h */
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
//  Implementation of the compression and decompression interface.
//--
*/

#ifndef __COMPRESSION_H__
#define __COMPRESSION_H__

#include "services/base.h"
#include "data_management/data/data_block.h"
#include "services/daal_defines.h"
#include "services/error_handling.h"

namespace daal
{
namespace data_management
{
/**
 * @ingroup data_compression
 * @{
 */
/**
 * <a name="DAAL-ENUM-DATA_MANAGEMENT__COMPRESSIONLEVEL"></a>
 * \brief %Compression levels    \DAAL_DEPRECATED
 */
enum CompressionLevel
{
    defaultLevel = -1, /*!< Default compression level */
    level0       = 0,  /*!< Minimum compression level, maximum speed */
    level1,            /*!< \n*/
    level2,            /*!< \n*/
    level3,            /*!< \n*/
    level4,            /*!< \n*/
    level5,            /*!< \n*/
    level6,            /*!< \n*/
    level7,            /*!< \n*/
    level8,            /*!< \n*/
    level9,            /*!< Maximum compression level, minimum speed */
    lastCompressionLevel = level9
};

/**
 * <a name="DAAL-ENUM-DATA_MANAGEMENT__COMPRESSIONMETHOD"></a>
 * \brief %Compression methods    \DAAL_DEPRECATED
 */
enum CompressionMethod
{
    zlib, /*!< DEFLATE compression method with a ZLIB block header or a simple GZIP block header */
    lzo,  /*!< LZO1X compatible compression method */
    rle,  /*!< Run-Length Encoding method */
    bzip2 /*!< BZIP2 compression method */
};

/**
 * \brief Contains version 1.0 of Intel(R) oneAPI Data Analytics Library interface.
 */
namespace interface1
{
/**
 * <a name="DAAL-STRUCT-DATA_MANAGEMENT__COMPRESSIONPARAMETER"></a>
 * \brief Parameters for compression and decompression    \DAAL_DEPRECATED
 *
 * \snippet compression/compression.h CompressionParameter source code
 *
 * \par Enumerations
 *      - \ref CompressionLevel - %Compression levels
 */
/* [CompressionParameter source code] */
struct DAAL_EXPORT CompressionParameter
{
    DAAL_DEPRECATED CompressionLevel level; /*!< Compression level */

    /**
     *  Default constructor
     *  \param[in] clevel   %Compression level, \ref CompressionLevel
     */
    DAAL_DEPRECATED CompressionParameter(CompressionLevel clevel = defaultLevel) : level(clevel) {}
};
/* [CompressionParameter source code] */

/**
 * <a name="DAAL-CLASS-DATA_MANAGEMENT__COMPRESSIONIFACE"></a>
 * \brief Abstract interface class for compression and decompression    \DAAL_DEPRECATED
 * <!-- \n<a href="DAAL-REF-COMPRESSION">Data compression usage model</a> -->
 */
class DAAL_EXPORT CompressionIface
{
public:
    /**
     * Associates an input data block with a compressor (or decompressor)
     * \param[in] inBlock Pointer to the data block to compress (or decompress). Must be at least size+offset bytes
     * \param[in] size     Number of bytes to compress (or decompress) in inBlock
     * \param[in] offset   Offset in bytes, the starting position for compression (or decompression) in inBlock
     */
    DAAL_DEPRECATED_VIRTUAL virtual void setInputDataBlock(byte * inBlock, size_t size, size_t offset) = 0;
    /**
     * Associates an input data block with a compressor (or decompressor)
     * \param[in] inBlock %DataBlock to compress (or decompress)
     */
    DAAL_DEPRECATED_VIRTUAL virtual void setInputDataBlock(DataBlock & inBlock) = 0;
    /**
     * Reports whether an output data block is full after a call to the run() method
     * \return True if an output data block is full, false otherwise
     */
    DAAL_DEPRECATED_VIRTUAL virtual bool isOutputDataBlockFull() = 0;
    /**
     * Returns the number of bytes used after a call to the run() method
     * \return Number of used bytes
     */
    DAAL_DEPRECATED_VIRTUAL virtual size_t getUsedOutputDataBlockSize() = 0;
    /**
     * Performs compression (or decompression) of a data block
     * \param[out] outBlock Pointer to the data block where compression (or decompression) results are stored. Must be at least size+offset bytes
     * \param[in] size       Number of bytes available in outBlock
     * \param[in] offset     Offset in bytes, the starting position for compression (or decompression) in outBlock
     */
    DAAL_DEPRECATED_VIRTUAL virtual void run(byte * outBlock, size_t size, size_t offset) = 0;
    /**
     * Performs compression (or decompression) of a data block
     * \param[out] outBlock %DataBlock where compression (or decompression) results are stored
     */
    DAAL_DEPRECATED_VIRTUAL virtual void run(DataBlock & outBlock) = 0;

    DAAL_DEPRECATED_VIRTUAL virtual ~CompressionIface() {}
};

/**
 * <a name="DAAL-CLASS-DATA_MANAGEMENT__COMPRESSION"></a>
 * \brief %Base class for compression and decompression    \DAAL_DEPRECATED
 * <!-- \n<a href="DAAL-REF-COMPRESSION">Data compression usage model</a> -->
 *
 * \par References
 *      - \ref services::ErrorCompressionNullInputStream "Data compression error codes"
 *      - CompressionParameter structure
 */
class DAAL_EXPORT Compression : public CompressionIface
{
public:
    /**
     * \brief Compression constructor
     */
    DAAL_DEPRECATED Compression() : _errors(new services::ErrorCollection())
    {
        this->_errors->setCanThrow(false);
        _isOutBlockFull   = false;
        _usedOutBlockSize = 0;
    }
    DAAL_DEPRECATED bool isOutputDataBlockFull() DAAL_C11_OVERRIDE { return _isOutBlockFull; }
    DAAL_DEPRECATED size_t getUsedOutputDataBlockSize() DAAL_C11_OVERRIDE { return _usedOutBlockSize; }
    DAAL_DEPRECATED_VIRTUAL virtual ~Compression() {}
    /**
     * Basic checks of input block parameters
     * \param[in] inBlock  Pointer to the input data block
     * \param[in] size     Size in bytes of the input data block
     */
    DAAL_DEPRECATED_VIRTUAL virtual void checkInputParams(byte * inBlock, size_t size)
    {
        if (inBlock == NULL)
        {
            this->_errors->add(services::ErrorCompressionNullInputStream);
        }
        if (size == 0)
        {
            this->_errors->add(services::ErrorCompressionEmptyInputStream);
        }
    }
    /**
     * Basic checks of output block parameters
     * \param[in] outBlock Pointer to output data block
     * \param[in] size      Size in bytes of the output data block
     */
    DAAL_DEPRECATED_VIRTUAL virtual void checkOutputParams(byte * outBlock, size_t size)
    {
        if (outBlock == NULL)
        {
            this->_errors->add(services::ErrorCompressionNullOutputStream);
        }
        if (size == 0)
        {
            this->_errors->add(services::ErrorCompressionEmptyOutputStream);
        }
    }

    DAAL_DEPRECATED services::SharedPtr<services::ErrorCollection> getErrors() const { return _errors; }

protected:
    bool _isOutBlockFull;
    size_t _usedOutBlockSize;

    services::SharedPtr<services::ErrorCollection> _errors;
};

/**
 * <a name="DAAL-CLASS-DATA_MANAGEMENT__COMPRESSORIMPL"></a>
 * \brief %Base class for the Compressor.    \DAAL_DEPRECATED
 * <!-- \n<a href="DAAL-REF-COMPRESSION">Data compression usage model</a> -->
 *
 * \par References
 *      - \ref services::ErrorCompressionNullInputStream "Data compression error codes"
 *      - CompressionParameter structure
 */
class DAAL_EXPORT CompressorImpl : public Compression
{
public:
    /**
     * \brief %Compressor constructor
     */
    DAAL_DEPRECATED CompressorImpl() : Compression() { _isInitialized = false; }
    DAAL_DEPRECATED_VIRTUAL virtual ~CompressorImpl() {}

protected:
    virtual void initialize() { _isInitialized = true; }
    bool _isInitialized;
};

/**
 * <a name="DAAL-CLASS-DATA_MANAGEMENT__DECOMPRESSORIMPL"></a>
 * \brief %Base class for the Decompressor.    \DAAL_DEPRECATED
 * <!-- \n<a href="DAAL-REF-COMPRESSION">Data compression usage model</a> -->
 *
 * \par References
 *      - \ref services::ErrorCompressionNullInputStream "Data compression error codes"
 *      - CompressionParameter structure
 */
class DAAL_EXPORT DecompressorImpl : public Compression
{
public:
    /**
     * \brief %Decompressor constructor
     */
    DAAL_DEPRECATED DecompressorImpl() : Compression() { _isInitialized = false; }
    DAAL_DEPRECATED_VIRTUAL virtual ~DecompressorImpl() {}

protected:
    virtual void initialize() { _isInitialized = true; }
    bool _isInitialized;
};

/**
 * <a name="DAAL-CLASS-DATA_MANAGEMENT__COMPRESSOR"></a>
 * \brief %Compressor class compresses an input data block and writes results into an output data block.    \DAAL_DEPRECATED
 * <!-- \n<a href="DAAL-REF-COMPRESSION">Data compression usage model</a> -->
 *
 * \tparam dcmethod Compression method, \ref CompressionMethod
 *
 * \par References
 *      - \ref services::ErrorCompressionNullInputStream "Data compression error codes"
 *      - CompressionParameter structure
 */
template <CompressionMethod dcmethod>
class DAAL_EXPORT Compressor : public CompressorImpl
{
public:
    /**
     * \brief %Compressor constructor
     */
    DAAL_DEPRECATED Compressor() : CompressorImpl() {}
    DAAL_DEPRECATED_VIRTUAL virtual ~Compressor() {}
};

/**
 * <a name="DAAL-CLASS-DATA_MANAGEMENT__DECOMPRESSOR"></a>
 * \brief %Decompressor class decompresses an input data block and writes results into an output data block.    \DAAL_DEPRECATED
 * <!-- \n<a href="DAAL-REF-COMPRESSION">Data compression usage model</a> -->
 *
 * \tparam dcmethod %Decompression method, \ref CompressionMethod
 *
 * \par References
 *      - \ref services::ErrorCompressionNullInputStream "Data compression error codes"
 *      - CompressionParameter structure
 */
template <CompressionMethod dcmethod>
class DAAL_EXPORT Decompressor : public DecompressorImpl
{
public:
    /**
     * \brief %Decompressor constructor
     */
    DAAL_DEPRECATED Decompressor() : DecompressorImpl() {}
    DAAL_DEPRECATED_VIRTUAL virtual ~Decompressor() {}
};
} // namespace interface1
using interface1::CompressionParameter;
using interface1::CompressionIface;
using interface1::Compression;
using interface1::CompressorImpl;
using interface1::DecompressorImpl;
using interface1::Compressor;
using interface1::Decompressor;
/** @} */

} // namespace data_management
} // namespace daal
#endif // __COMPRESSION_H
