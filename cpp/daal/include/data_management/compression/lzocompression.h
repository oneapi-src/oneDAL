/* file: lzocompression.h */
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
//  Implementation of the LZO1X_11 compression and decompression interface.
//--
*/

#ifndef __LZOCOMPRESSION_H__
#define __LZOCOMPRESSION_H__
#include "data_management/compression/compression.h"

namespace daal
{
namespace data_management
{
namespace interface1
{
class DAAL_EXPORT LzoCompressionParameter : public data_management::CompressionParameter
{
public:
    DAAL_DEPRECATED LzoCompressionParameter(size_t _preHeadBytes = 0, size_t _postHeadBytes = 0)
        : data_management::CompressionParameter(defaultLevel), preHeadBytes(_preHeadBytes), postHeadBytes(_postHeadBytes)
    {}
    DAAL_DEPRECATED ~LzoCompressionParameter() {}

    DAAL_DEPRECATED size_t preHeadBytes;  /*!< Size in bytes of section 1 of the LZO compressed block header */
    DAAL_DEPRECATED size_t postHeadBytes; /*!< Size in bytes of section 4 of the LZO compressed block header */
};

template <>
class DAAL_EXPORT Compressor<lzo> : public data_management::CompressorImpl
{
public:
    DAAL_DEPRECATED Compressor();
    DAAL_DEPRECATED ~Compressor();

    DAAL_DEPRECATED void setInputDataBlock(byte * inBlock, size_t size, size_t offset);

    DAAL_DEPRECATED void setInputDataBlock(DataBlock & inBlock) { setInputDataBlock(inBlock.getPtr(), inBlock.getSize(), 0); }

    DAAL_DEPRECATED void run(byte * outBlock, size_t size, size_t offset);

    DAAL_DEPRECATED void run(DataBlock & outBlock) { run(outBlock.getPtr(), outBlock.getSize(), 0); }

    DAAL_DEPRECATED LzoCompressionParameter parameter; /*!< LZO compression parameters structure */

protected:
    void initialize();

private:
    void * _next_in;
    size_t _avail_in;
    void * _next_out;
    size_t _avail_out;
    void * _p_lzo_state;
    size_t _preHeadBytes;
    size_t _postHeadBytes;

    void finalizeCompression();
};

template <>
class DAAL_EXPORT Decompressor<lzo> : public data_management::DecompressorImpl
{
public:
    DAAL_DEPRECATED Decompressor();
    DAAL_DEPRECATED ~Decompressor();

    DAAL_DEPRECATED void setInputDataBlock(byte * inBlock, size_t size, size_t offset);

    DAAL_DEPRECATED void setInputDataBlock(DataBlock & inBlock) { return setInputDataBlock(inBlock.getPtr(), inBlock.getSize(), 0); }

    DAAL_DEPRECATED void run(byte * outBlock, size_t size, size_t offset);

    DAAL_DEPRECATED void run(DataBlock & outBlock) { run(outBlock.getPtr(), outBlock.getSize(), 0); }

    DAAL_DEPRECATED LzoCompressionParameter parameter; /*!< LZO compression parameters structure */

protected:
    void initialize();

private:
    void * _next_in;
    size_t _avail_in;
    void * _next_out;
    size_t _avail_out;
    void * _p_lzo_state;
    size_t _preHeadBytes;
    size_t _postHeadBytes;

    void * _internalBuff;
    size_t _internalBuffOff;
    size_t _internalBuffLen;

    void finalizeCompression();
};
} // namespace interface1
using interface1::LzoCompressionParameter;
using interface1::Compressor;
using interface1::Decompressor;

} //namespace data_management
} //namespace daal
#endif //__LZOCOMPRESSION_H
