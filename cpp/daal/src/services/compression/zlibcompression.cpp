/* file: zlibcompression.cpp */
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
//  Implementation of ZLIB DEFLATE (de-)compression method.
//--
*/

#ifndef DAAL_REF

    #include "data_management/compression/zlibcompression.h"
    #include "data_management/compression/lzocompression.h"
    #include "services/daal_memory.h"

namespace daal
{
namespace data_management
{
Compressor<zlib>::Compressor() : data_management::CompressorImpl()
{
    _strmp                  = (void *)new Compressor<lzo>();
    this->_isOutBlockFull   = ((Compressor<lzo> *)_strmp)->isOutputDataBlockFull();
    this->_usedOutBlockSize = ((Compressor<lzo> *)_strmp)->getUsedOutputDataBlockSize();
    this->_errors           = ((Compressor<lzo> *)_strmp)->getErrors();
}

void Compressor<zlib>::initialize() {}

Compressor<zlib>::~Compressor()
{
    delete (Compressor<lzo> *)_strmp;
}

void Compressor<zlib>::finalizeCompression() {}

void Compressor<zlib>::resetCompression() {}

void Compressor<zlib>::setInputDataBlock(byte * in, size_t len, size_t off)
{
    ((Compressor<lzo> *)_strmp)->setInputDataBlock(in, len, off);
    this->_isOutBlockFull   = ((Compressor<lzo> *)_strmp)->isOutputDataBlockFull();
    this->_usedOutBlockSize = ((Compressor<lzo> *)_strmp)->getUsedOutputDataBlockSize();
    this->_errors           = ((Compressor<lzo> *)_strmp)->getErrors();
}

void Compressor<zlib>::run(byte * out, size_t outLen, size_t off)
{
    ((Compressor<lzo> *)_strmp)->run(out, outLen, off);
    this->_isOutBlockFull   = ((Compressor<lzo> *)_strmp)->isOutputDataBlockFull();
    this->_usedOutBlockSize = ((Compressor<lzo> *)_strmp)->getUsedOutputDataBlockSize();
    this->_errors           = ((Compressor<lzo> *)_strmp)->getErrors();
}

Decompressor<zlib>::Decompressor() : data_management::DecompressorImpl()
{
    _strmp                  = (void *)new Decompressor<lzo>();
    this->_isOutBlockFull   = ((Decompressor<lzo> *)_strmp)->isOutputDataBlockFull();
    this->_usedOutBlockSize = ((Decompressor<lzo> *)_strmp)->getUsedOutputDataBlockSize();
    this->_errors           = ((Decompressor<lzo> *)_strmp)->getErrors();
}

void Decompressor<zlib>::initialize() {}

Decompressor<zlib>::~Decompressor()
{
    delete (Decompressor<lzo> *)_strmp;
}

void Decompressor<zlib>::finalizeCompression() {}

void Decompressor<zlib>::resetCompression() {}

void Decompressor<zlib>::setInputDataBlock(byte * in, size_t len, size_t off)
{
    ((Decompressor<lzo> *)_strmp)->setInputDataBlock(in, len, off);
    this->_isOutBlockFull   = ((Decompressor<lzo> *)_strmp)->isOutputDataBlockFull();
    this->_usedOutBlockSize = ((Decompressor<lzo> *)_strmp)->getUsedOutputDataBlockSize();
    this->_errors           = ((Decompressor<lzo> *)_strmp)->getErrors();
}

void Decompressor<zlib>::run(byte * out, size_t outLen, size_t off)
{
    ((Decompressor<lzo> *)_strmp)->run(out, outLen, off);
    this->_isOutBlockFull   = ((Decompressor<lzo> *)_strmp)->isOutputDataBlockFull();
    this->_usedOutBlockSize = ((Decompressor<lzo> *)_strmp)->getUsedOutputDataBlockSize();
    this->_errors           = ((Decompressor<lzo> *)_strmp)->getErrors();
}
} //namespace data_management
} //namespace daal

#endif
