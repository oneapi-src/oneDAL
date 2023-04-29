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
enum DAAL_DEPRECATED CompressionLevel
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

enum DAAL_DEPRECATED CompressionMethod
{
    zlib, /*!< DEFLATE compression method with a ZLIB block header or a simple GZIP block header */
    lzo,  /*!< LZO1X compatible compression method */
    rle,  /*!< Run-Length Encoding method */
    bzip2 /*!< BZIP2 compression method */
};

namespace interface1
{
struct DAAL_EXPORT CompressionParameter
{
    DAAL_DEPRECATED CompressionLevel level; /*!< Compression level */

    DAAL_DEPRECATED CompressionParameter(CompressionLevel clevel = defaultLevel) : level(clevel) {}
};

class DAAL_EXPORT CompressionIface
{
public:
    DAAL_DEPRECATED_VIRTUAL virtual void setInputDataBlock(byte * inBlock, size_t size, size_t offset) = 0;

    DAAL_DEPRECATED_VIRTUAL virtual void setInputDataBlock(DataBlock & inBlock) = 0;

    DAAL_DEPRECATED_VIRTUAL virtual bool isOutputDataBlockFull() = 0;

    DAAL_DEPRECATED_VIRTUAL virtual size_t getUsedOutputDataBlockSize() = 0;

    DAAL_DEPRECATED_VIRTUAL virtual void run(byte * outBlock, size_t size, size_t offset) = 0;

    DAAL_DEPRECATED_VIRTUAL virtual void run(DataBlock & outBlock) = 0;

    DAAL_DEPRECATED_VIRTUAL virtual ~CompressionIface() {}
};

class DAAL_EXPORT Compression : public CompressionIface
{
public:
    DAAL_DEPRECATED Compression() : _errors(new services::ErrorCollection())
    {
        this->_errors->setCanThrow(false);
        _isOutBlockFull   = false;
        _usedOutBlockSize = 0;
    }
    DAAL_DEPRECATED bool isOutputDataBlockFull() DAAL_C11_OVERRIDE { return _isOutBlockFull; }
    DAAL_DEPRECATED size_t getUsedOutputDataBlockSize() DAAL_C11_OVERRIDE { return _usedOutBlockSize; }
    DAAL_DEPRECATED_VIRTUAL virtual ~Compression() {}

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

class DAAL_EXPORT CompressorImpl : public Compression
{
public:
    DAAL_DEPRECATED CompressorImpl() : Compression() { _isInitialized = false; }
    DAAL_DEPRECATED_VIRTUAL virtual ~CompressorImpl() {}

protected:
    virtual void initialize() { _isInitialized = true; }
    bool _isInitialized;
};

class DAAL_EXPORT DecompressorImpl : public Compression
{
public:
    DAAL_DEPRECATED DecompressorImpl() : Compression() { _isInitialized = false; }
    DAAL_DEPRECATED_VIRTUAL virtual ~DecompressorImpl() {}

protected:
    virtual void initialize() { _isInitialized = true; }
    bool _isInitialized;
};

template <CompressionMethod dcmethod>
class DAAL_EXPORT Compressor : public CompressorImpl
{
public:
    DAAL_DEPRECATED Compressor() : CompressorImpl() {}
    DAAL_DEPRECATED_VIRTUAL virtual ~Compressor() {}
};

template <CompressionMethod dcmethod>
class DAAL_EXPORT Decompressor : public DecompressorImpl
{
public:
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

} // namespace data_management
} // namespace daal
#endif // __COMPRESSION_H
