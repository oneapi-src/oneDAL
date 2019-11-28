/* file: lzocompression.cpp */
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
//  Implementation of LZO1X_11 (de-)compression method.
//--
*/

#include "lzocompression.h"
#include "ipp.h"
#include "daal_memory.h"

#if defined(_MSC_VER)
    #define EXPECT(x, y) (x)
#else
    #define EXPECT(x, y) (__builtin_expect((x), (y)))
#endif

#define BLOCK_HEADER_BYTES 8

namespace daal
{
namespace data_management
{
Compressor<lzo>::Compressor() : data_management::CompressorImpl()
{
    Ipp32u state_size;

    _preHeadBytes  = parameter.preHeadBytes;
    _postHeadBytes = parameter.postHeadBytes;

    _next_in     = NULL;
    _avail_in    = 0;
    _next_out    = NULL;
    _avail_out   = 0;
    _p_lzo_state = NULL;

    ippInit();
    int errCode = ippsEncodeLZOGetSize(IppLZO1XST, 0, &state_size);
    if (errCode != ippStsNoErr)
    {
        this->_errors->add(services::ErrorLzoInternal);
    }
    _p_lzo_state = (void *)daal::services::daal_calloc((size_t)state_size);
    if (_p_lzo_state == NULL)
    {
        this->_errors->add(services::ErrorMemoryAllocationFailed);
    }

    errCode = ippsEncodeLZOInit_8u(IppLZO1XST, 0, (IppLZOState_8u *)_p_lzo_state);
    if (errCode != ippStsNoErr)
    {
        daal::services::daal_free(_p_lzo_state);
        _p_lzo_state = NULL;
        this->_errors->add(services::ErrorLzoInternal);
    }

    switch (errCode)
    {
    case ippStsBadArgErr:
    case ippStsNullPtrErr:
        finalizeCompression();
        this->_errors->add(services::ErrorLzoInternal);
        return;
    default: break;
    }

    if (this->_errors->size() != 0)
    {
        finalizeCompression();
        return;
    }

    _isInitialized = false;
}

void Compressor<lzo>::initialize()
{
    _preHeadBytes  = parameter.preHeadBytes;
    _postHeadBytes = parameter.postHeadBytes;
    _isInitialized = true;
}

Compressor<lzo>::~Compressor()
{
    if (_p_lzo_state) daal::services::daal_free(_p_lzo_state);
    _p_lzo_state = NULL;
}

void Compressor<lzo>::finalizeCompression()
{
    daal::services::daal_free(_p_lzo_state);
    _p_lzo_state          = NULL;
    this->_isOutBlockFull = 0;
    _next_in              = NULL;
    _avail_in             = 0;
    _next_out             = NULL;
    _avail_out            = 0;
}

void Compressor<lzo>::setInputDataBlock(byte * in, size_t len, size_t off)
{
    if (_isInitialized == false)
    {
        initialize();
    }

    checkInputParams(in, len);
    if (this->_errors->size() != 0)
    {
        return;
    }

    _avail_in = len;
    _next_in  = in + off;
}

void Compressor<lzo>::run(byte * out, size_t outLen, size_t off)
{
    if (_isInitialized == false)
    {
        this->_errors->add(services::ErrorLzoInternal);
        return;
    }

    Ipp32u tmp_avail_in;
    Ipp32u tmp_avail_out;

    checkOutputParams(out, outLen);
    if (this->_errors->size() != 0)
    {
        finalizeCompression();
        return;
    }

    _avail_out              = outLen;
    _next_out               = out + off;
    this->_isOutBlockFull   = 0;
    this->_usedOutBlockSize = 0;

    if (_avail_out < (67 + BLOCK_HEADER_BYTES + _preHeadBytes + _postHeadBytes) + 2)
    {
        finalizeCompression();
        this->_errors->add(services::ErrorLzoOutputStreamSizeIsNotEnough);
        return;
    }

    if (_avail_out < _avail_in + (_avail_in) / 16 + 67 + BLOCK_HEADER_BYTES + _preHeadBytes + _postHeadBytes)
    {
        tmp_avail_in = ((_avail_out - (67 + BLOCK_HEADER_BYTES + _preHeadBytes + _postHeadBytes)) * 16) / 17;
    }
    else
    {
        tmp_avail_in = _avail_in;
    }

    tmp_avail_out = _avail_out - (BLOCK_HEADER_BYTES + _preHeadBytes + _postHeadBytes);
    int errCode   = ippsEncodeLZO_8u((const Ipp8u *)(_next_in), tmp_avail_in,
                                   (Ipp8u *)((byte *)(_next_out) + (BLOCK_HEADER_BYTES + _preHeadBytes + _postHeadBytes)), &tmp_avail_out,
                                   (IppLZOState_8u *)_p_lzo_state);
    if (errCode != ippStsNoErr)
    {
        finalizeCompression();
        this->_errors->add(services::ErrorLzoInternal);
        return;
    }
    _avail_out                                         = _avail_out - tmp_avail_out - (BLOCK_HEADER_BYTES + _preHeadBytes + _postHeadBytes);
    ((Ipp32u *)((byte *)_next_out + _preHeadBytes))[0] = (Ipp32u)tmp_avail_in;
    ((Ipp32u *)((byte *)_next_out + _preHeadBytes))[1] = (Ipp32u)tmp_avail_out;
    this->_usedOutBlockSize += (BLOCK_HEADER_BYTES + _preHeadBytes + _postHeadBytes);
    this->_usedOutBlockSize += tmp_avail_out;
    _avail_in = _avail_in - tmp_avail_in;
    if (_avail_in > 0)
    {
        _next_in              = (void *)((byte *)(_next_in) + tmp_avail_in);
        this->_isOutBlockFull = 1;
    }
}

Decompressor<lzo>::Decompressor() : data_management::DecompressorImpl()
{
    _next_in              = NULL;
    _avail_in             = 0;
    _next_out             = NULL;
    _avail_out            = 0;
    this->_isOutBlockFull = 0;
    _internalBuff         = NULL;
    _internalBuffOff      = 0;
    _internalBuffLen      = 0;

    _preHeadBytes  = parameter.preHeadBytes;
    _postHeadBytes = parameter.postHeadBytes;

    ippInit();
    _isInitialized = false;
}

void Decompressor<lzo>::initialize()
{
    _preHeadBytes  = parameter.preHeadBytes;
    _postHeadBytes = parameter.postHeadBytes;
    _isInitialized = true;
}

Decompressor<lzo>::~Decompressor()
{
    if (_internalBuff != NULL)
    {
        daal::services::daal_free(_internalBuff);
        _internalBuff = NULL;
    }
}

void Decompressor<lzo>::finalizeCompression()
{
    if (_internalBuff != NULL)
    {
        daal::services::daal_free(_internalBuff);
    }
    _internalBuff    = NULL;
    _internalBuffLen = 0;
    _internalBuffOff = 0;
}

void Decompressor<lzo>::setInputDataBlock(byte * in, size_t len, size_t off)
{
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

    if (len <= BLOCK_HEADER_BYTES + _preHeadBytes + _postHeadBytes)
    {
        finalizeCompression();
        this->_errors->add(services::ErrorLzoDataFormatLessThenHeader);
        return;
    }

    _avail_in = len;
    _next_in  = in + off;
}

void Decompressor<lzo>::run(byte * out, size_t outLen, size_t off)
{
    if (_isInitialized == false)
    {
        this->_errors->add(services::ErrorLzoInternal);
        return;
    }

    Ipp32u tmp_avail_out         = 0;
    Ipp32u uncompressedBlockSize = 0;
    Ipp32u compressedBlockSize   = 0;
    this->_isOutBlockFull        = 0;
    this->_usedOutBlockSize      = 0;
    int result                   = 0;

    checkOutputParams(out, outLen);
    if (this->_errors->size() != 0)
    {
        finalizeCompression();
        return;
    }

    _avail_out = outLen;
    _next_out  = out + off;
    if (_internalBuffLen - _internalBuffOff > 0)
    {
        if (_avail_out < _internalBuffLen - _internalBuffOff)
        {
            result |= daal::services::internal::daal_memcpy_s((void *)(_next_out), _avail_out, (void *)(((byte *)_internalBuff) + _internalBuffOff),
                                                              _avail_out);
            if (result)
            {
                this->_errors->add(services::ErrorMemoryCopyFailedInternal);
                return;
            }

            _internalBuffOff += _avail_out;
            this->_usedOutBlockSize += _avail_out;
            _avail_out            = 0;
            this->_isOutBlockFull = 1;
            return;
        }
        else
        {
            result |=
                daal::services::internal::daal_memcpy_s((void *)(_next_out), _internalBuffLen - _internalBuffOff,
                                                        (void *)(((byte *)_internalBuff) + _internalBuffOff), _internalBuffLen - _internalBuffOff);
            if (result)
            {
                this->_errors->add(services::ErrorMemoryCopyFailedInternal);
                return;
            }

            this->_usedOutBlockSize += _internalBuffLen - _internalBuffOff;
            _avail_out = _avail_out - (_internalBuffLen - _internalBuffOff);
            _next_out  = (void *)((byte *)_next_out + (_internalBuffLen - _internalBuffOff));
            daal::services::daal_free(_internalBuff);
            _internalBuff    = NULL;
            _internalBuffLen = 0;
            _internalBuffOff = 0;
            if (_avail_in == 0)
            {
                return;
            }
        }
    }

    do
    {
        if (EXPECT(_avail_in < BLOCK_HEADER_BYTES + _preHeadBytes + _postHeadBytes, 0))
        {
            finalizeCompression();
            this->_errors->add(services::ErrorLzoDataFormatLessThenHeader);
            return;
        }

        uncompressedBlockSize = ((Ipp32u *)((byte *)_next_in + _preHeadBytes))[0];
        compressedBlockSize   = ((Ipp32u *)((byte *)_next_in + _preHeadBytes))[1];

        if (EXPECT(_avail_in < compressedBlockSize + BLOCK_HEADER_BYTES + _preHeadBytes + _postHeadBytes, 0))
        {
            finalizeCompression();
            this->_errors->add(services::ErrorLzoDataFormatNotFullBlock);
            return;
        }
        if (_avail_out < uncompressedBlockSize)
        {
            _internalBuff = daal::services::daal_calloc(uncompressedBlockSize);
            if (EXPECT(_internalBuff == NULL, 0))
            {
                finalizeCompression();
                this->_errors->add(services::ErrorMemoryAllocationFailed);
                return;
            }
            _internalBuffLen = uncompressedBlockSize;
            _internalBuffOff = 0;
            tmp_avail_out    = uncompressedBlockSize;
            int errCode      = ippsDecodeLZO_8u((const Ipp8u *)((byte *)_next_in + BLOCK_HEADER_BYTES + _preHeadBytes + _postHeadBytes),
                                           compressedBlockSize, (Ipp8u *)((byte *)_internalBuff), &tmp_avail_out);
            if (EXPECT(errCode != ippStsNoErr, 0))
            {
                finalizeCompression();
                switch (errCode)
                {
                case ippStsSrcSizeLessExpected: this->_errors->add(services::ErrorLzoDataFormat); return;
                default: this->_errors->add(services::ErrorLzoInternal); return;
                }
            }

            result |= daal::services::internal::daal_memcpy_s((void *)(_next_out), _avail_out, (void *)(((byte *)_internalBuff) + _internalBuffOff),
                                                              _avail_out);
            if (result)
            {
                this->_errors->add(services::ErrorMemoryCopyFailedInternal);
                return;
            }

            _internalBuffOff += _avail_out;
            this->_usedOutBlockSize += _avail_out;
            _avail_out            = 0;
            this->_isOutBlockFull = 1;
            _avail_in             = _avail_in - (compressedBlockSize + BLOCK_HEADER_BYTES + _preHeadBytes + _postHeadBytes);
            if (_avail_in > 0)
            {
                _next_in = (void *)((byte *)(_next_in) + compressedBlockSize + BLOCK_HEADER_BYTES + _preHeadBytes + _postHeadBytes);
            }
            return;
        }
        tmp_avail_out = _avail_out;
        int errCode   = ippsDecodeLZO_8u((const Ipp8u *)((byte *)_next_in + BLOCK_HEADER_BYTES + _preHeadBytes + _postHeadBytes), compressedBlockSize,
                                       (Ipp8u *)((byte *)_next_out), &tmp_avail_out);
        if (EXPECT(errCode != ippStsNoErr, 0))
        {
            finalizeCompression();
            switch (errCode)
            {
            case ippStsSrcSizeLessExpected: this->_errors->add(services::ErrorLzoDataFormat); return;
            default: this->_errors->add(services::ErrorLzoInternal); return;
            }
        }
        _avail_in = _avail_in - (compressedBlockSize + BLOCK_HEADER_BYTES + _preHeadBytes + _postHeadBytes);
        if (_avail_in > 0)
        {
            _next_in = (void *)((byte *)(_next_in) + compressedBlockSize + BLOCK_HEADER_BYTES + _preHeadBytes + _postHeadBytes);
        }
        _avail_out = _avail_out - tmp_avail_out;
        _next_out  = (byte *)_next_out + tmp_avail_out;
        this->_usedOutBlockSize += tmp_avail_out;
    } while (_avail_in > 0 && _avail_out > 0);

    if (_avail_in > 0)
    {
        this->_isOutBlockFull = 1;
    }
}
} //namespace data_management
} //namespace daal
