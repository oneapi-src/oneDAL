/* file: rlecompression.h */
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
class DAAL_EXPORT RleCompressionParameter : public data_management::CompressionParameter
{
public:
    DAAL_DEPRECATED RleCompressionParameter(bool _isBlockHeader = 1)
        : data_management::CompressionParameter(defaultLevel), isBlockHeader(_isBlockHeader)
    {}

    DAAL_DEPRECATED ~RleCompressionParameter() {}

    DAAL_DEPRECATED bool isBlockHeader; /*!< RLE block header presence flag. True if a RLE block header is present, false otherwise */
};

template <>
class DAAL_EXPORT Compressor<rle> : public data_management::CompressorImpl
{
public:
    DAAL_DEPRECATED Compressor();
    DAAL_DEPRECATED ~Compressor();

    DAAL_DEPRECATED void setInputDataBlock(byte * inBlock, size_t size, size_t offset);

    DAAL_DEPRECATED void setInputDataBlock(DataBlock & inBlock) { setInputDataBlock(inBlock.getPtr(), inBlock.getSize(), 0); }

    DAAL_DEPRECATED void run(byte * outBlock, size_t size, size_t offset);

    DAAL_DEPRECATED void run(DataBlock & outBlock) { run(outBlock.getPtr(), outBlock.getSize(), 0); }

    DAAL_DEPRECATED RleCompressionParameter parameter; /*!< RLE compression parameters structure */

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

template <>
class DAAL_EXPORT Decompressor<rle> : public data_management::DecompressorImpl
{
public:
    DAAL_DEPRECATED Decompressor();
    DAAL_DEPRECATED ~Decompressor();

    DAAL_DEPRECATED void setInputDataBlock(byte * inBlock, size_t size, size_t offset);

    DAAL_DEPRECATED void setInputDataBlock(DataBlock & inBlock) { setInputDataBlock(inBlock.getPtr(), inBlock.getSize(), 0); }

    DAAL_DEPRECATED void run(byte * outBlock, size_t size, size_t offset);

    DAAL_DEPRECATED void run(DataBlock & outBlock) { run(outBlock.getPtr(), outBlock.getSize(), 0); }

    DAAL_DEPRECATED RleCompressionParameter parameter; /*!< RLE compression parameters structure */

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
} // namespace interface1
using interface1::RleCompressionParameter;
using interface1::Compressor;
using interface1::Decompressor;

} //namespace data_management
} //namespace daal
#endif //__RLECOMPRESSION_H
