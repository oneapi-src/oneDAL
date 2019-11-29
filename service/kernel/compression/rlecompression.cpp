/* file: rlecompression.cpp */
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
//  Implementation of Run Length Encoding interface.
//--
*/

#include "rlecompression.h"
#include "ipp.h"
#include "daal_memory.h"

#if defined(_MSC_VER)
    #define EXPECT(x, y) (x)
#else
    #define EXPECT(x, y) (__builtin_expect((x), (y)))
#endif

namespace daal
{
namespace data_management
{
Compressor<rle>::Compressor() : data_management::CompressorImpl()
{
    _headBytes = 8;
    if (!(parameter.isBlockHeader))
    {
        _headBytes = 0;
    }
    _next_in   = NULL;
    _avail_in  = 0;
    _next_out  = NULL;
    _avail_out = 0;

    ippInit();
    _isInitialized = false;
}

void Compressor<rle>::initialize()
{
    if (!(parameter.isBlockHeader))
    {
        _headBytes = 0;
    }
    _isInitialized = true;
}

Compressor<rle>::~Compressor() {}

void Compressor<rle>::finalizeCompression()
{
    _next_in   = NULL;
    _avail_in  = 0;
    _next_out  = NULL;
    _avail_out = 0;
}

void Compressor<rle>::setInputDataBlock(byte * in, size_t len, size_t off)
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

    _avail_in = len;
    _next_in  = in + off;
}

void Compressor<rle>::run(byte * out, size_t outLen, size_t off)
{
    if (_isInitialized == false)
    {
        this->_errors->add(services::ErrorRleInternal);
        return;
    }

    int tmp_avail_in  = 0;
    int tmp_avail_out = 0;

    checkOutputParams(out, outLen);
    if (this->_errors->size() != 0)
    {
        finalizeCompression();
        return;
    }

    if (outLen <= _headBytes)
    {
        finalizeCompression();
        this->_errors->add(services::ErrorRleOutputStreamSizeIsNotEnough);
        return;
    }

    _avail_out            = outLen;
    _next_out             = out + off;
    this->_isOutBlockFull = 0;
    tmp_avail_in          = _avail_in;
    tmp_avail_out         = _avail_out - _headBytes;

    int errCode = ippsEncodeRLE_8u((Ipp8u **)&_next_in, &tmp_avail_in, (Ipp8u *)((byte *)_next_out + _headBytes), &tmp_avail_out);

    switch (errCode)
    {
    case ippStsSizeErr:
    case ippStsNullPtrErr:
        finalizeCompression();
        this->_errors->add(services::ErrorRleInternal);
        return;
    case ippStsDstSizeLessExpected:
        _avail_out -= (tmp_avail_out + _headBytes);
        this->_usedOutBlockSize  = tmp_avail_out + _headBytes;
        this->_isOutBlockFull    = 1;
        ((Ipp32u *)_next_out)[0] = _avail_in - tmp_avail_in;
        ((Ipp32u *)_next_out)[1] = tmp_avail_out;
        _avail_in                = tmp_avail_in;
        return;
    default: break;
    }
    ((Ipp32u *)_next_out)[0] = _avail_in - tmp_avail_in;
    ((Ipp32u *)_next_out)[1] = tmp_avail_out;
    this->_usedOutBlockSize  = tmp_avail_out + _headBytes;
    finalizeCompression();
}

Decompressor<rle>::Decompressor() : data_management::DecompressorImpl()
{
    _headBytes = 8;
    if (!(parameter.isBlockHeader))
    {
        _headBytes = 0;
    }

    _internalBuff    = NULL;
    _internalBuffLen = 0;
    _internalBuffOff = 0;

    _next_in   = NULL;
    _avail_in  = 0;
    _next_out  = NULL;
    _avail_out = 0;

    ippInit();
    _isInitialized = false;
}

void Decompressor<rle>::initialize()
{
    if (!(parameter.isBlockHeader))
    {
        _headBytes = 0;
    }
    _isInitialized = true;
}
Decompressor<rle>::~Decompressor()
{
    if (_internalBuff != NULL)
    {
        daal::services::daal_free(_internalBuff);
        _internalBuff = NULL;
    }
}

void Decompressor<rle>::finalizeCompression()
{
    if (_internalBuff != NULL)
    {
        daal::services::daal_free(_internalBuff);
    }
    _internalBuff    = NULL;
    _internalBuffLen = 0;
    _internalBuffOff = 0;

    _next_in   = NULL;
    _avail_in  = 0;
    _next_out  = NULL;
    _avail_out = 0;
}

void Decompressor<rle>::setInputDataBlock(byte * in, size_t len, size_t off)
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

    _avail_in = len;
    _next_in  = in + off;
}

void Decompressor<rle>::run(byte * out, size_t outLen, size_t off)
{
    if (_isInitialized == false)
    {
        this->_errors->add(services::ErrorRleInternal);
        return;
    }

    int tmp_avail_in  = 0;
    int tmp_avail_out = 0;

    Ipp32u uncompressedBlockSize = 0;
    Ipp32u compressedBlockSize   = 0;

    Ipp8u * tmp_in_ptr = NULL;

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
    tmp_avail_in            = _avail_in;
    tmp_avail_out           = _avail_out;
    int result              = 0;

    if (_headBytes > 0)
    {
        if (_internalBuffLen - _internalBuffOff > 0)
        {
            if (_avail_out < _internalBuffLen - _internalBuffOff)
            {
                result |= daal::services::internal::daal_memcpy_s((void *)(_next_out), _avail_out,
                                                                  (void *)(((byte *)_internalBuff) + _internalBuffOff), _avail_out);
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
                result |= daal::services::internal::daal_memcpy_s((void *)(_next_out), _internalBuffLen - _internalBuffOff,
                                                                  (void *)(((byte *)_internalBuff) + _internalBuffOff),
                                                                  _internalBuffLen - _internalBuffOff);
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
            if (EXPECT(_avail_in < _headBytes, 0))
            {
                finalizeCompression();
                this->_errors->add(services::ErrorRleDataFormatLessThenHeader);
                return;
            }

            uncompressedBlockSize = ((Ipp32u *)((byte *)_next_in))[0];
            compressedBlockSize   = ((Ipp32u *)((byte *)_next_in))[1];
            _next_in              = (void *)((byte *)_next_in + _headBytes);
            tmp_avail_in          = compressedBlockSize;

            if (EXPECT(_avail_in < compressedBlockSize + _headBytes, 0))
            {
                finalizeCompression();
                this->_errors->add(services::ErrorRleDataFormatNotFullBlock);
                return;
            }
            if (_avail_out < uncompressedBlockSize)
            {
                _internalBuff = daal::services::daal_calloc(uncompressedBlockSize);
                if (EXPECT(_internalBuff == NULL, 0))
                {
                    this->_errors->add(services::ErrorMemoryAllocationFailed);
                    return;
                }
                _internalBuffLen = uncompressedBlockSize;
                _internalBuffOff = 0;
                tmp_avail_out    = uncompressedBlockSize;
                tmp_in_ptr       = (Ipp8u *)_next_in;

                int errCode = ippsDecodeRLE_8u((Ipp8u **)&tmp_in_ptr, &tmp_avail_in, (Ipp8u *)_internalBuff, &tmp_avail_out);
                if (EXPECT(errCode != ippStsNoErr, 0))
                {
                    finalizeCompression();
                    switch (errCode)
                    {
                    case ippStsSrcDataErr: this->_errors->add(services::ErrorRleDataFormat); return;
                    case ippStsSizeErr:
                    case ippStsDstSizeLessExpected:
                    default: this->_errors->add(services::ErrorRleInternal); return;
                    }
                }

                result |= daal::services::internal::daal_memcpy_s((void *)(_next_out), _avail_out,
                                                                  (void *)(((byte *)_internalBuff) + _internalBuffOff), _avail_out);
                if (result)
                {
                    this->_errors->add(services::ErrorMemoryCopyFailedInternal);
                    return;
                }

                _internalBuffOff += _avail_out;
                this->_usedOutBlockSize += _avail_out;
                _avail_out            = 0;
                this->_isOutBlockFull = 1;
                if (EXPECT(tmp_avail_in > compressedBlockSize, 0))
                {
                    finalizeCompression();
                    this->_errors->add(services::ErrorRleInternal);
                    return;
                }
                _avail_in = _avail_in - (compressedBlockSize - tmp_avail_in + _headBytes);
                _next_in  = ((byte *)_next_in + (compressedBlockSize - tmp_avail_in));
                return;
            }

            tmp_avail_out = _avail_out;
            tmp_in_ptr    = (Ipp8u *)_next_in;
            int errCode   = ippsDecodeRLE_8u((Ipp8u **)&tmp_in_ptr, &tmp_avail_in, (Ipp8u *)_next_out, &tmp_avail_out);
            if (EXPECT(errCode != ippStsNoErr, 0))
            {
                finalizeCompression();
                switch (errCode)
                {
                case ippStsSrcDataErr: this->_errors->add(services::ErrorRleDataFormat); return;
                case ippStsSizeErr:
                case ippStsDstSizeLessExpected:
                default: this->_errors->add(services::ErrorRleInternal); return;
                }
            }

            if (EXPECT(tmp_avail_in > compressedBlockSize, 0))
            {
                finalizeCompression();
                this->_errors->add(services::ErrorRleInternal);
                return;
            }
            _avail_in = _avail_in - (compressedBlockSize - tmp_avail_in + _headBytes);
            _next_in  = ((byte *)_next_in + (compressedBlockSize - tmp_avail_in));

            _avail_out = _avail_out - tmp_avail_out;
            _next_out  = (byte *)_next_out + tmp_avail_out;
            this->_usedOutBlockSize += tmp_avail_out;

        } while (_avail_in > 0 && _avail_out > 0);

        if (_avail_in > 0)
        {
            this->_isOutBlockFull = 1;
        }
        return;
    }
    else
    {
        int errCode = ippsDecodeRLE_8u((Ipp8u **)&_next_in, &tmp_avail_in, (Ipp8u *)_next_out, &tmp_avail_out);

        switch (errCode)
        {
        case ippStsSizeErr:
            finalizeCompression();
            this->_errors->add(services::ErrorRleInternal);
            return;
        case ippStsSrcDataErr:
            finalizeCompression();
            this->_errors->add(services::ErrorRleDataFormat);
            return;
        case ippStsDstSizeLessExpected:
            _avail_out -= tmp_avail_out;
            this->_usedOutBlockSize = tmp_avail_out;
            this->_isOutBlockFull   = 1;
            _avail_in               = tmp_avail_in;
            return;
        default: break;
        }
    }

    this->_usedOutBlockSize = tmp_avail_out;
    finalizeCompression();
}

} //namespace data_management
} //namespace daal
