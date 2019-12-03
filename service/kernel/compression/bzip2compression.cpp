/* file: bzip2compression.cpp */
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
//  Implementation of BZip2 (de-)compression interface.
//--
*/

#include "daal_bzlib.h"
#include "bzip2compression.h"
#include "ipp.h"
#include "daal_memory.h"

/*workaround, use STD BZIP2 while Intel(R) IPP doesn't work,
remove 1 after fix */
#define CompressInit   BZ2_bzCompressInit
#define Compress       BZ2_bzCompress
#define CompressEnd    BZ2_bzCompressEnd
#define DecompressInit BZ2_bzDecompressInit1
#define Decompress     BZ2_bzDecompress1
#define DecompressEnd  BZ2_bzDecompressEnd1
/* end of workaround */

namespace daal
{
namespace data_management
{
void Compressor<bzip2>::checkBZipError(int error)
{
    switch (error)
    {
    case BZ_PARAM_ERROR:
        finalizeCompression();
        this->_errors->add(services::ErrorBzip2Parameters);
        return;
    case BZ_MEM_ERROR:
        finalizeCompression();
        this->_errors->add(services::ErrorBzip2MemoryAllocationFailed);
        return;
    case BZ_CONFIG_ERROR:
        finalizeCompression();
        this->_errors->add(services::ErrorBzip2Internal);
        return;
    default: break;
    }
}

void Decompressor<bzip2>::checkBZipError(int error)
{
    switch (error)
    {
    case BZ_PARAM_ERROR:
        finalizeCompression();
        this->_errors->add(services::ErrorBzip2Parameters);
        return;
    case BZ_MEM_ERROR:
        finalizeCompression();
        this->_errors->add(services::ErrorBzip2MemoryAllocationFailed);
        return;
    case BZ_CONFIG_ERROR:
        finalizeCompression();
        this->_errors->add(services::ErrorBzip2Internal);
        return;
    default: break;
    }
}

Compressor<bzip2>::Compressor() : data_management::CompressorImpl()
{
    _strmp = NULL;
    _strmp = (void *)daal::services::daal_calloc(sizeof(bz_stream));
    if (_strmp == NULL)
    {
        this->_errors->add(services::ErrorMemoryAllocationFailed);
        return;
    }

    ((bz_stream *)_strmp)->bzalloc = NULL;
    ((bz_stream *)_strmp)->bzfree  = NULL;
    ((bz_stream *)_strmp)->opaque  = NULL;
    _flush                         = BZ_RUN;

    _blockSize100k = parameter.level;
    if (_blockSize100k == defaultLevel)
    {
        _blockSize100k = level9;
    }
    if (_blockSize100k == level0)
    {
        _blockSize100k = level1;
    }

    int errCode = CompressInit((bz_stream *)_strmp, _blockSize100k, 0, 0);
    checkBZipError(errCode);

    ((bz_stream *)_strmp)->avail_in  = 0;
    ((bz_stream *)_strmp)->next_in   = NULL;
    ((bz_stream *)_strmp)->avail_out = 0;
    ((bz_stream *)_strmp)->next_out  = NULL;

    this->_isOutBlockFull   = 0;
    this->_usedOutBlockSize = 0;

    _comprLen        = 0;
    _comprLenLeft    = 0;
    _comprBlockThres = _blockSize100k * 1024 * 97;

    _isInitialized = false;
}

void Compressor<bzip2>::initialize()
{
    _blockSize100k = parameter.level;
    if (_blockSize100k == defaultLevel)
    {
        _blockSize100k = level9;
    }
    if (_blockSize100k == level0)
    {
        _blockSize100k = level1;
    }
    resetCompression();
    _isInitialized = true;
}

Compressor<bzip2>::~Compressor()
{
    (void)CompressEnd(((bz_stream *)_strmp));
    if (_strmp) daal::services::daal_free(((bz_stream *)_strmp));
    _strmp = NULL;
}

void Compressor<bzip2>::finalizeCompression()
{
    (void)CompressEnd(((bz_stream *)_strmp));
    this->_isOutBlockFull            = 0;
    this->_usedOutBlockSize          = 0;
    ((bz_stream *)_strmp)->avail_in  = 0;
    ((bz_stream *)_strmp)->next_in   = NULL;
    ((bz_stream *)_strmp)->avail_out = 0;
    ((bz_stream *)_strmp)->next_out  = NULL;
    _flush                           = BZ_RUN;
}

void Compressor<bzip2>::resetCompression()
{
    (void)CompressEnd(((bz_stream *)_strmp));
    this->_isOutBlockFull = 0;

    int errCode = CompressInit((bz_stream *)_strmp, _blockSize100k, 0, 0);
    checkBZipError(errCode);

    ((bz_stream *)_strmp)->avail_in  = 0;
    ((bz_stream *)_strmp)->next_in   = NULL;
    ((bz_stream *)_strmp)->avail_out = 0;
    ((bz_stream *)_strmp)->next_out  = NULL;

    _comprLen        = 0;
    _comprLenLeft    = 0;
    _comprBlockThres = _blockSize100k * 1024 * 97;

    _flush = BZ_RUN;
}

void Compressor<bzip2>::setInputDataBlock(byte * in, size_t len, size_t off)
{
    if (this->_errors->size() != 0)
    {
        return;
    }

    if (_isInitialized == false)
    {
        initialize();
    }

    checkInputParams(in, len);
    if (this->_errors->size() != 0)
    {
        finalizeCompression();
        return;
    }

    _comprLen     = len;
    _comprLenLeft = len;
    _startAddr    = in + off;

    ((bz_stream *)_strmp)->avail_in = _comprLenLeft > _comprBlockThres ? _comprBlockThres : _comprLenLeft;
    ((bz_stream *)_strmp)->next_in  = (char *)(_startAddr);
}

void Compressor<bzip2>::run(byte * out, size_t outLen, size_t off)
{
    if (_isInitialized == false)
    {
        this->_errors->add(services::ErrorBzip2Internal);
    }

    checkOutputParams(out, outLen);
    if (this->_errors->size() != 0)
    {
        finalizeCompression();
        return;
    }

    ((bz_stream *)_strmp)->avail_out = outLen;
    ((bz_stream *)_strmp)->next_out  = (char *)(out + off);
    this->_isOutBlockFull            = 0;
    this->_usedOutBlockSize          = 0;

    if (_comprLen > _comprBlockThres)
    {
        if (_flush != BZ_FINISH) _flush = BZ_FLUSH;

        do
        {
            int errCode = Compress(((bz_stream *)_strmp), _flush);
            checkBZipError(errCode);

            switch (errCode)
            {
            case BZ_RUN_OK:
            {
                size_t processedSize = _comprLenLeft > _comprBlockThres ? _comprBlockThres : _comprLenLeft;
                _comprLenLeft -= processedSize;

                if (_comprLenLeft /*((bz_stream *)_strmp)->avail_in*/ == 0)
                {
                    _flush = BZ_FINISH;
                }
                else
                {
                    size_t sizeToCompress           = _comprLenLeft > _comprBlockThres ? _comprBlockThres : _comprLenLeft;
                    ((bz_stream *)_strmp)->avail_in = sizeToCompress;
                    ((bz_stream *)_strmp)->next_in  = (char *)(_startAddr + _comprLen - _comprLenLeft);
                }
            }
            break;
            case BZ_STREAM_END: //normal termination, reset and return
                this->_usedOutBlockSize = outLen - ((bz_stream *)_strmp)->avail_out;
                resetCompression();
                return;
            case BZ_FINISH_OK: //need more output
                if (((bz_stream *)_strmp)->avail_out == 0)
                {
                    this->_usedOutBlockSize = outLen - ((bz_stream *)_strmp)->avail_out;
                    this->_isOutBlockFull   = 1;
                    return;
                }
                break;
            case BZ_FLUSH_OK:
                if (((bz_stream *)_strmp)->avail_out == 0)
                {
                    this->_usedOutBlockSize = outLen - ((bz_stream *)_strmp)->avail_out;
                    this->_isOutBlockFull   = 1;
                    return;
                }
                break;
            default: finalizeCompression(); this->_errors->add(services::ErrorBzip2Internal);
            }
        } while (((bz_stream *)_strmp)->avail_out > 0);

        this->_usedOutBlockSize = outLen;
        this->_isOutBlockFull   = 1;
    }
    else
    {
        do
        {
            int errCode = Compress(((bz_stream *)_strmp), _flush);
            checkBZipError(errCode);

            switch (errCode)
            {
            case BZ_RUN_OK:
                if (((bz_stream *)_strmp)->avail_in == 0)
                {
                    _flush = BZ_FINISH;
                }
                break;
            case BZ_STREAM_END: //normal termination, reset and return
                this->_usedOutBlockSize = outLen - ((bz_stream *)_strmp)->avail_out;
                resetCompression();
                return;
            case BZ_FINISH_OK: //need more output
                if (((bz_stream *)_strmp)->avail_out == 0)
                {
                    this->_usedOutBlockSize = outLen - ((bz_stream *)_strmp)->avail_out;
                    this->_isOutBlockFull   = 1;
                    return;
                }
                break;
            default: finalizeCompression(); this->_errors->add(services::ErrorBzip2Internal);
            }
        } while (((bz_stream *)_strmp)->avail_out > 0);

        this->_usedOutBlockSize = outLen;
        this->_isOutBlockFull   = 1;
    }
}

Decompressor<bzip2>::Decompressor() : data_management::DecompressorImpl()
{
    _strmp = NULL;
    _strmp = (void *)daal::services::daal_calloc(sizeof(bz_stream));
    if (_strmp == NULL)
    {
        this->_errors->add(services::ErrorMemoryAllocationFailed);
    }

    ((bz_stream *)_strmp)->bzalloc = NULL;
    ((bz_stream *)_strmp)->bzfree  = NULL;
    ((bz_stream *)_strmp)->opaque  = NULL;

    int errCode = DecompressInit((bz_stream *)_strmp, 0, 0);
    checkBZipError(errCode);

    ((bz_stream *)_strmp)->avail_in  = 0;
    ((bz_stream *)_strmp)->next_in   = NULL;
    ((bz_stream *)_strmp)->avail_out = 0;
    ((bz_stream *)_strmp)->next_out  = NULL;

    if (errCode != BZ_OK && this->_errors->size() == 0)
    {
        this->_errors->add(services::ErrorBzip2Internal);
        return;
    }
    this->_isOutBlockFull = 0;
    _isInitialized        = true;
}

Decompressor<bzip2>::~Decompressor()
{
    (void)DecompressEnd(((bz_stream *)_strmp));
    daal::services::daal_free(((bz_stream *)_strmp));
    _strmp = NULL;
}

void Decompressor<bzip2>::finalizeCompression()
{
    (void)DecompressEnd(((bz_stream *)_strmp));
    this->_isOutBlockFull            = 0;
    ((bz_stream *)_strmp)->avail_in  = 0;
    ((bz_stream *)_strmp)->next_in   = NULL;
    ((bz_stream *)_strmp)->avail_out = 0;
    ((bz_stream *)_strmp)->next_out  = NULL;
}

void Decompressor<bzip2>::resetCompression()
{
    (void)DecompressEnd(((bz_stream *)_strmp));
    this->_isOutBlockFull = 0;

    int errCode = DecompressInit((bz_stream *)_strmp, 0, 0);
    checkBZipError(errCode);

    ((bz_stream *)_strmp)->avail_in  = 0;
    ((bz_stream *)_strmp)->next_in   = NULL;
    ((bz_stream *)_strmp)->avail_out = 0;
    ((bz_stream *)_strmp)->next_out  = NULL;
}

void Decompressor<bzip2>::initialize()
{
    _isInitialized = true;
}

void Decompressor<bzip2>::setInputDataBlock(byte * in, size_t len, size_t off)
{
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

    ((bz_stream *)_strmp)->avail_in = len;
    ((bz_stream *)_strmp)->next_in  = (char *)(in + off);
}

void Decompressor<bzip2>::run(byte * out, size_t outLen, size_t off)
{
    checkOutputParams(out, outLen);
    if (this->_errors->size() != 0)
    {
        finalizeCompression();
        return;
    }

    ((bz_stream *)_strmp)->avail_out = outLen;
    ((bz_stream *)_strmp)->next_out  = (char *)(out + off);
    this->_isOutBlockFull            = 0;
    this->_usedOutBlockSize          = 0;
    do
    {
        int errCode = Decompress((bz_stream *)_strmp);

        switch (errCode)
        {
        case BZ_STREAM_END: //normal termination, reset and return
            this->_usedOutBlockSize = outLen - ((bz_stream *)_strmp)->avail_out;
            this->_isOutBlockFull   = 0;
            if (((bz_stream *)_strmp)->avail_in > 0)
            {
                byte * tmpPtrIn   = (byte *)((bz_stream *)_strmp)->next_in;
                size_t tmpSizeIn  = ((bz_stream *)_strmp)->avail_in;
                byte * tmpPtrOut  = (byte *)((bz_stream *)_strmp)->next_out;
                size_t tmpSizeOut = ((bz_stream *)_strmp)->avail_out;
                resetCompression();
                setInputDataBlock(tmpPtrIn, tmpSizeIn, 0);
                ((bz_stream *)_strmp)->next_out  = (char *)tmpPtrOut;
                ((bz_stream *)_strmp)->avail_out = tmpSizeOut;
            }
            else
            {
                resetCompression();
            }
            break;
        case BZ_OK: //need more output or input
            this->_usedOutBlockSize = outLen - ((bz_stream *)_strmp)->avail_out;
            this->_isOutBlockFull   = 0;
            if (((bz_stream *)_strmp)->avail_out < 1)
            {
                this->_isOutBlockFull = 1;
            }
            return;
        case BZ_DATA_ERROR:
        case BZ_DATA_ERROR_MAGIC: finalizeCompression(); this->_errors->add(services::ErrorBzip2DataFormat);
        case BZ_PARAM_ERROR: finalizeCompression(); this->_errors->add(services::ErrorBzip2Parameters);
        case BZ_MEM_ERROR: finalizeCompression(); this->_errors->add(services::ErrorBzip2MemoryAllocationFailed);
        default: finalizeCompression(); this->_errors->add(services::ErrorBzip2Internal);
        }
    } while (((bz_stream *)_strmp)->avail_in > 0);
}
} //namespace data_management
} //namespace daal
