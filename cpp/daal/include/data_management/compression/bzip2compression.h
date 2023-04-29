/* file: bzip2compression.h */
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
//  Implementation of the bzip2 compression and decompression interface.
//--
*/

#ifndef __BZIP2COMPRESSION_H__
#define __BZIP2COMPRESSION_H__
#include "data_management/compression/compression.h"

namespace daal
{
namespace data_management
{
namespace interface1
{
class DAAL_EXPORT Bzip2CompressionParameter : public data_management::CompressionParameter
{
public:
    DAAL_DEPRECATED Bzip2CompressionParameter(CompressionLevel clevel = defaultLevel) : data_management::CompressionParameter(clevel) {}
    DAAL_DEPRECATED ~Bzip2CompressionParameter() {}
};

template <>
class DAAL_EXPORT Compressor<bzip2> : public data_management::CompressorImpl
{
public:
    DAAL_DEPRECATED Compressor();
    DAAL_DEPRECATED ~Compressor();

    DAAL_DEPRECATED void setInputDataBlock(byte * inBlock, size_t size, size_t offset);

    DAAL_DEPRECATED void setInputDataBlock(DataBlock & inBlock) { return setInputDataBlock(inBlock.getPtr(), inBlock.getSize(), 0); }

    DAAL_DEPRECATED void run(byte * outBlock, size_t size, size_t offset);

    DAAL_DEPRECATED void run(DataBlock & outBlock) { run(outBlock.getPtr(), outBlock.getSize(), 0); }

    DAAL_DEPRECATED Bzip2CompressionParameter parameter; /*!< Bzip2 compression parameters structure */

protected:
    void initialize();

private:
    void * _strmp;
    int _flush;
    int _blockSize100k;
    size_t _comprLen;
    size_t _comprLenLeft;
    size_t _comprBlockThres;
    byte * _startAddr;

    void finalizeCompression();
    void resetCompression();
    void checkBZipError(int error);
};

template <>
class DAAL_EXPORT Decompressor<bzip2> : public data_management::DecompressorImpl
{
public:
    DAAL_DEPRECATED Decompressor();
    DAAL_DEPRECATED ~Decompressor();

    DAAL_DEPRECATED void setInputDataBlock(byte * inBlock, size_t size, size_t offset);

    DAAL_DEPRECATED void setInputDataBlock(DataBlock & inBlock) { setInputDataBlock(inBlock.getPtr(), inBlock.getSize(), 0); }

    DAAL_DEPRECATED void run(byte * outBlock, size_t size, size_t offset);

    DAAL_DEPRECATED void run(DataBlock & outBlock) { run(outBlock.getPtr(), outBlock.getSize(), 0); }

    DAAL_DEPRECATED Bzip2CompressionParameter parameter; /*!< Bzip2 compression parameters structure */

protected:
    void initialize();

private:
    void * _strmp;

    void finalizeCompression();
    void resetCompression();
    void checkBZipError(int error);
};
} // namespace interface1
using interface1::Bzip2CompressionParameter;
using interface1::Compressor;
using interface1::Decompressor;

} //namespace data_management
} //namespace daal
#endif //__BZIP2COMPRESSION_H
