/* file: zlibcompression.cpp */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation
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

#include "daal_zlib.h"
#include "zlibcompression.h"
#include "daal_memory.h"

namespace daal
{
namespace data_management
{
Compressor<zlib>::Compressor() : data_management::CompressorImpl()
{
    this->_isOutBlockFull = 0;
    _strmp                = NULL;
    _isInitialized        = false;
}

void Compressor<zlib>::initialize()
{
    int _windowBits;

    if (parameter.gzHeader)
    {
        _windowBits = 31;
    }
    else
    {
        _windowBits = 15;
    }

    int _memLevel;
    int _strategy;
    int _method;

    _memLevel = 8;
    _strategy = Z_DEFAULT_STRATEGY;
    _method   = 8; //Z_DEFLATED;

    _strmp = NULL;
    _strmp = (void *)daal::services::daal_calloc(sizeof(z_stream));
    if (_strmp == NULL)
    {
        this->_errors->add(services::ErrorMemoryAllocationFailed);
        return;
    }

    ((z_stream *)_strmp)->zalloc = Z_NULL;
    ((z_stream *)_strmp)->zfree  = Z_NULL;
    ((z_stream *)_strmp)->opaque = Z_NULL;
    _flush                       = Z_SYNC_FLUSH;

    int errCode = deflateInit2(((z_stream *)_strmp), (int)(parameter.level), _method, _windowBits, _memLevel, _strategy);

    _isInitialized = true;

    ((z_stream *)_strmp)->avail_in  = 0;
    ((z_stream *)_strmp)->next_in   = Z_NULL;
    ((z_stream *)_strmp)->avail_out = 0;
    ((z_stream *)_strmp)->next_out  = Z_NULL;

    if (!((errCode == Z_OK) || (errCode == Z_STREAM_END) || (errCode == Z_BUF_ERROR)))
    {
        switch (errCode)
        {
        case Z_STREAM_ERROR:
            finalizeCompression();
            this->_errors->add(services::ErrorZlibParameters);
            return;
        case Z_MEM_ERROR:
            finalizeCompression();
            this->_errors->add(services::ErrorZlibMemoryAllocationFailed);
            return;
        case Z_VERSION_ERROR:
        default: this->_errors->add(services::ErrorZlibInternal); return;
        }
    }
}

Compressor<zlib>::~Compressor()
{
    (void)deflateEnd(((z_stream *)_strmp));
    if (_strmp) daal::services::daal_free(_strmp);
    _strmp = NULL;
}

void Compressor<zlib>::finalizeCompression()
{
    (void)deflateEnd(((z_stream *)_strmp));
    _flush = Z_SYNC_FLUSH;
}

void Compressor<zlib>::resetCompression()
{
    (void)deflateReset(((z_stream *)_strmp));
    _flush = Z_SYNC_FLUSH;
}

void Compressor<zlib>::setInputDataBlock(byte * in, size_t len, size_t off)
{
    if (_isInitialized == false)
    {
        initialize();
    }

    if (this->_errors->size() != 0)
    {
        return;
    }

    checkInputParams(in, len);
    if (this->_errors->size() != 0)
    {
        finalizeCompression();
        return;
    }

    ((z_stream *)_strmp)->avail_in = len;
    ((z_stream *)_strmp)->next_in  = in + off;
}

void Compressor<zlib>::run(byte * out, size_t outLen, size_t off)
{
    if (_isInitialized == false)
    {
        this->_errors->add(services::ErrorZlibInternal);
        return;
    }

    checkOutputParams(out, outLen);
    if (this->_errors->size() != 0)
    {
        finalizeCompression();
        return;
    }

    ((z_stream *)_strmp)->avail_out = outLen;
    ((z_stream *)_strmp)->next_out  = out + off;
    this->_isOutBlockFull           = 0;

    int errCode = deflate(((z_stream *)_strmp), _flush);

    switch (errCode)
    {
    case Z_STREAM_END:
        this->_usedOutBlockSize = outLen - ((z_stream *)_strmp)->avail_out;
        this->_isOutBlockFull   = 0;
        resetCompression();
        return;
    case Z_STREAM_ERROR:
        finalizeCompression();
        this->_errors->add(services::ErrorZlibInternal);
        return;
    case Z_OK:
    case Z_BUF_ERROR:
        if ((((z_stream *)_strmp)->avail_in == 0) || _flush == Z_FINISH)
        {
            _flush = Z_FINISH;
        }
        if (((z_stream *)_strmp)->avail_out != 0 && ((z_stream *)_strmp)->avail_in == 0)
        {
            errCode = deflate(((z_stream *)_strmp), _flush);
            if (errCode != Z_STREAM_END)
            {
                finalizeCompression();
                this->_errors->add(services::ErrorZlibInternal);
                return;
            }
            this->_usedOutBlockSize = outLen - ((z_stream *)_strmp)->avail_out;
            this->_isOutBlockFull   = 0;
            resetCompression();
        }
        else
        {
            this->_usedOutBlockSize = outLen - ((z_stream *)_strmp)->avail_out;
            this->_isOutBlockFull   = 1;
        }
        return;
    default:
        finalizeCompression();
        this->_errors->add(services::ErrorZlibInternal);
        return;
    }
}

Decompressor<zlib>::Decompressor() : data_management::DecompressorImpl()
{
    _strmp = NULL;
    _strmp = daal::services::daal_calloc(sizeof(z_stream));
    if (_strmp == NULL)
    {
        this->_errors->add(services::ErrorMemoryAllocationFailed);
        return;
    }

    ((z_stream *)_strmp)->zalloc = Z_NULL;
    ((z_stream *)_strmp)->zfree  = Z_NULL;
    ((z_stream *)_strmp)->opaque = Z_NULL;
    _flush                       = Z_SYNC_FLUSH;

    if (this->_errors->size() != 0)
    {
        return;
    }
    this->_isOutBlockFull = 0;

    _isInitialized = false;
}

void Decompressor<zlib>::initialize()
{
    int _windowBits;

    if (parameter.gzHeader)
    {
        _windowBits = 31;
    }
    else
    {
        _windowBits = 15;
    }

    ((z_stream *)_strmp)->avail_in  = 0;
    ((z_stream *)_strmp)->next_in   = Z_NULL;
    int errCode                     = inflateInit2(((z_stream *)_strmp), _windowBits);
    ((z_stream *)_strmp)->avail_out = 0;
    ((z_stream *)_strmp)->next_out  = Z_NULL;

    _isInitialized = true;

    if (!((errCode == Z_OK) || (errCode == Z_STREAM_END)))
    {
        switch (errCode)
        {
        case Z_STREAM_ERROR:
            finalizeCompression();
            this->_isOutBlockFull = 0;
            this->_errors->add(services::ErrorZlibParameters);
            return;
        case Z_MEM_ERROR:
            finalizeCompression();
            this->_isOutBlockFull = 0;
            this->_errors->add(services::ErrorZlibMemoryAllocationFailed);
            return;
        case Z_VERSION_ERROR:
        default: this->_errors->add(services::ErrorZlibInternal); return;
        }
    }
}

Decompressor<zlib>::~Decompressor()
{
    (void)inflateEnd(((z_stream *)_strmp));
    daal::services::daal_free(_strmp);
    _strmp = NULL;
}

void Decompressor<zlib>::finalizeCompression()
{
    (void)inflateEnd(((z_stream *)_strmp));
}

void Decompressor<zlib>::resetCompression()
{
    (void)inflateReset(((z_stream *)_strmp));
}

void Decompressor<zlib>::setInputDataBlock(byte * in, size_t len, size_t off)
{
    if (_isInitialized == false)
    {
        initialize();
    }

    if (this->_errors->size() != 0)
    {
        return;
    }

    checkInputParams(in, len);
    if (this->_errors->size() != 0)
    {
        finalizeCompression();
        return;
    }

    ((z_stream *)_strmp)->avail_in = len;
    ((z_stream *)_strmp)->next_in  = in + off;
}

void Decompressor<zlib>::run(byte * out, size_t outLen, size_t off)
{
    if (_isInitialized == false)
    {
        this->_errors->add(services::ErrorZlibInternal);
        return;
    }

    checkOutputParams(out, outLen);
    if (this->_errors->size() != 0)
    {
        finalizeCompression();
        return;
    }

    ((z_stream *)_strmp)->avail_out = outLen;
    ((z_stream *)_strmp)->next_out  = out + off;

    do
    {
        int errCode = inflate(((z_stream *)_strmp), Z_NO_FLUSH);
        switch (errCode)
        {
        case Z_STREAM_END:
            this->_usedOutBlockSize = outLen - ((z_stream *)_strmp)->avail_out;
            this->_isOutBlockFull   = 0;
            if (((z_stream *)_strmp)->avail_in > 0)
            {
                byte * tmpPtrIn   = (byte *)((z_stream *)_strmp)->next_in;
                size_t tmpSizeIn  = ((z_stream *)_strmp)->avail_in;
                byte * tmpPtrOut  = (byte *)((z_stream *)_strmp)->next_out;
                size_t tmpSizeOut = ((z_stream *)_strmp)->avail_out;
                resetCompression();
                setInputDataBlock(tmpPtrIn, tmpSizeIn, 0);
                ((z_stream *)_strmp)->next_out  = tmpPtrOut;
                ((z_stream *)_strmp)->avail_out = tmpSizeOut;
            }
            break;
        case Z_NEED_DICT:
            finalizeCompression();
            this->_isOutBlockFull = 0;
            this->_errors->add(services::ErrorZlibNeedDictionary);
            return;
        case Z_DATA_ERROR:
            finalizeCompression();
            this->_isOutBlockFull = 0;
            this->_errors->add(services::ErrorZlibDataFormat);
            return;
        case Z_MEM_ERROR:
            finalizeCompression();
            this->_isOutBlockFull = 0;
            this->_errors->add(services::ErrorZlibMemoryAllocationFailed);
            return;
        case Z_BUF_ERROR:
        case Z_STREAM_ERROR:
            finalizeCompression();
            this->_isOutBlockFull = 0;
            this->_errors->add(services::ErrorZlibInternal);
            return;
        case Z_OK:
            this->_usedOutBlockSize = outLen - ((z_stream *)_strmp)->avail_out;
            if (((z_stream *)_strmp)->avail_in > 0 && ((z_stream *)_strmp)->avail_out == 0)
            {
                this->_isOutBlockFull = 1;
                return;
            }
            else
            {
                this->_isOutBlockFull = 0;
                if (((z_stream *)_strmp)->avail_in > 0)
                {
                    byte * tmpPtrIn   = (byte *)((z_stream *)_strmp)->next_in;
                    size_t tmpSizeIn  = ((z_stream *)_strmp)->avail_in;
                    byte * tmpPtrOut  = (byte *)((z_stream *)_strmp)->next_out;
                    size_t tmpSizeOut = ((z_stream *)_strmp)->avail_out;
                    resetCompression();
                    setInputDataBlock(tmpPtrIn, tmpSizeIn, 0);
                    ((z_stream *)_strmp)->next_out  = tmpPtrOut;
                    ((z_stream *)_strmp)->avail_out = tmpSizeOut;
                }
            }
            break;
        default:
            finalizeCompression();
            this->_isOutBlockFull = 0;
            this->_errors->add(services::ErrorZlibInternal);
            return;
        }
    } while (((z_stream *)_strmp)->avail_in > 0);
}
} //namespace data_management
} //namespace daal
