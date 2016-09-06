/*******************************************************************************
* Copyright 2015-2016 Intel Corporation
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

#if !defined( __IPPFPK_H__ )
#define __IPPFPK_H__

#ifdef __cplusplus
extern "C" {
#endif

#include "ippdefs.h"
#include "ippfpk_types.h"
#include "ippfpk_redefs.h"
#include "ippversion.h"

/* /////////////////////////////////////////////////////////////////////////////
//  Name:       ippInit
//  Purpose:    Automatic switching to best for current cpu library code using.
//  Returns:
//   ippStsNoErr
//
//  Parameter:  nothing
//
//  Notes:      At the moment of this function execution no any other IPP function
//              has to be working
*/
IPPAPI( IppStatus, ippInit, ( void ))

/* /////////////////////////////////////////////////////////////////////////////
//  Name:               ippsAdler32
//  Purpose:            Computes the adler32(ITUT V.42) checksum for the source vector.
//
//  Parameters:
//   pSrc               Pointer to the source vector.
//   srcLen             Length of the source vector.
//   pAdler32           Pointer to the checksum value.
//  Return:
//   ippStsNoErr        Indicates no error.
//   ippStsNullPtrErr   Indicates an error when the pSrc pointer is NULL.
//   ippStsSizeErr      Indicates an error when the length of the source vector is less
//                      than or equal to zero.
//
*/
IPPAPI( IppStatus, ippsAdler32_8u, (const Ipp8u* pSrc, int srcLen, Ipp32u* pAdler32) )

/* /////////////////////////////////////////////////////////////////////////////
//  Name:               ippsCRC32
//  Purpose:            Computes the CRC32(ITUT V.42) checksum for the source vector.
//
//  Parameters:
//   pSrc               Pointer to the source vector.
//   srcLen             Length of the source vector.
//   pCRC32             Pointer to the checksum value.
//  Return:
//   ippStsNoErr        Indicates no error.
//   ippStsNullPtrErr   Indicates an error when the pSrc pointer is NULL.
//   ippStsSizeErr      Indicates an error when the length of the source vector is less
//                      than or equal to zero.
//
*/
IPPAPI( IppStatus, ippsCRC32_8u, (const Ipp8u* pSrc, int srcLen, Ipp32u* pCRC32) )

/* /////////////////////////////////////////////////////////////////////////////
//  Name:               ippsInflateBuildHuffTable
//  Purpose:            Builds literal/length and distance Huffman code table for
//                      decoding a block that was compressed with usage dynamic Huffman codes
//                      according to the "deflate" format (rfc1951)
//
//  Parameters:
//    pCodeLens         Pointer to the common array with literal/length and distance
//                      Huffman code lengths
//    nLitCodeLens      Number of literal/length Huffman code lengths
//    nDistCodeLens     Number of distance Huffman code lengths
//  Return:
//    ippStsNullPtrErr  One or several pointer(s) is NULL
//    ippStsSizeErr     nLitCodeLens is greater than 286, or nLitCodeLens is greater than 30
//                      (according to rfc1951)
//    ippStsSrcDataErr  Invalid literal/length and distance set has been met
//                      in the common lengths array
//    ippStsNoErr       No errors
//
*/
IPPAPI(IppStatus, ippsInflateBuildHuffTable, ( const Ipp16u* pCodeLens,
                                               unsigned int nLitCodeLens,
                                               unsigned int nDistCodeLens,
                                               IppInflateState *pIppInflateState ))

/* /////////////////////////////////////////////////////////////////////////////
//  Name:               ippsInflate_8u
//  Purpose:            Decodes of the "deflate" format (rfc1951)
//                      according to the type of Huffman code tables
//
//  Parameters:
//    ppSrc             Double pointer to the source vector
//    pSrcLen           Pointer to the length of the source vector
//    pCode             Pointer to the bit buffer
//    pCodeLenBits      Number of valid bits in the bit buffer
//    winIdx            Index of the sliding window start position
//    ppDst             Double pointer to the destination vector
//    pDstLen           Pointer to the length of the destination vector
//    dstIdx            Index of the current position in the destination vector
//    pMode             Pointer to the current decode mode
//    pIppInflateState  Pointer to the structure that contains decode parameters
//  Return:
//    ippStsNullPtrErr  One or several pointer(s) is NULL
//    ippStsSizeErr     codeLenBits is greater than 32, or
//                      winIdx is greater than pIppInflateState->winSize, or
//                      dstIdx is greater than dstLen
//    ippStsSrcDataErr  Invalid literal/length and distance set has been met
//                      during decoding
//    ippStsNoErr       No errors
//
*/
IPPAPI(IppStatus, ippsInflate_8u, ( Ipp8u** ppSrc, unsigned int* pSrcLen,
                                    Ipp32u* pCode, unsigned int* pCodeLenBits,
                                    unsigned int winIdx,
                                    Ipp8u** ppDst, unsigned int* pDstLen, unsigned int dstIdx,
                                    IppInflateMode* pMode, IppInflateState *pIppInflateState ))

/* /////////////////////////////////////////////////////////////////////////////
//  Name:               ippsDeflateLZ77_8u
//  Purpose:            Perform LZ77 encoding according to
//                      the compression level
//
//  Parameters:
//    ppSrc             Double pointer to the source vector
//    pSrcLen           Pointer to the length of the source vector
//    pSrcIdx           Pointer to the index of the current position in
//                      the source vector. This parameter is used by
//                      the function for correlation current possition of
//                      the source vector and indexes in the hash tables.
//                      The normalization of this index and the hash tables
//                      must only be done every 2GB of the source data
//                      instead of 64K (the zlib approach)
//    pWindow           Pointer to the sliding window, which is used as
//                      the dictionary for LZ77 encoding
//    winSize           Size of the window and the hash prev table
//    pHashHead         Pointer to heads of the hash chains. This table is
//                      initialized with (-winSize) value for correct processing
//                      of the first bytes of the source vector
//    pHashPrev         Pointer to links to older strings with the same
//                      hash index
//    hashSize          Size of the hash head table
//    pLitFreqTable     Pointer to the literals/lengths frequency table
//    pDistFreqTable    Pointer to the distances frequency table
//    pLitDst           Pointer to the literals/lengths destination vector
//    pDistDst          Pointer to the distances destination vector
//    pDstLen           Pointer to the length of the destination vectors
//    comprLevel        Compression level. It is like the zlib compression level
//    flush             Flush value
//  Return:
//    ippStsNullPtrErr  One or several pointer(s) is NULL
//    ippStsNoErr       No errors
//
*/

IPPAPI( IppStatus, ippsDeflateLZ77_8u, (
                     const Ipp8u** ppSrc, Ipp32u* pSrcLen, Ipp32u* pSrcIdx,
                     const Ipp8u* pWindow, Ipp32u winSize,
                     Ipp32s* pHashHead, Ipp32s* pHashPrev, Ipp32u hashSize,
                     IppDeflateFreqTable pLitFreqTable[286],
                     IppDeflateFreqTable pDistFreqTable[30],
                     Ipp8u* pLitDst, Ipp16u* pDistDst, Ipp32u* pDstLen,
                     int comprLevel, IppLZ77Flush flush ) )

/* /////////////////////////////////////////////////////////////////////////////
//  Name:              ippsDeflateHuff_8u
//  Purpose:           Performs Huffman encoding
//
//  Parameters:
//    pLitSrc          Pointer to the literals/lengths source vector
//    pDistSrc         Pointer to the distances source vector
//    pSrcLen          Pointer to the length of the source vectors
//    pCode            Pointer to the bit buffer
//    pCodeLenBits     Pointer to the number of valid bits in the bit buffer
//    pLitHuffCodes    Pointer to the literals/lengths Huffman codes
//    pDistHuffCodes   Pointer to the distances Huffman codes
//    pDst             Pointer to the destination vector
//    pDstIdx          Pointer to the index in the destination vector, the zlib
//                     uses the knowingly sufficient intermediate buffer for
//                     the Huffman encoding, so we need to know indexes of
//                     the first (input parameter) and the last (output parameter)
//                     symbols, which are written by the function
//  Return:
//    ippStsNullPtrErr One or several pointer(s) is NULL
//    ippStsNoErr      No errors
//
*/
IPPAPI( IppStatus, ippsDeflateHuff_8u, (
          const Ipp8u* pLitSrc, const Ipp16u* pDistSrc, Ipp32u srcLen,
          Ipp16u* pCode, Ipp32u* pCodeLenBits,
          IppDeflateHuffCode pLitHuffCodes[286],
          IppDeflateHuffCode pDistHuffCodes[30],
          Ipp8u* pDst, Ipp32u* pDstIdx ) )

/**************************************************
IPP LZO Definitions
***************************************************/

/*******************************************************************/

/* /////////////////////////////////////////////////////////////////////////////
//  Name:       ippsEncodeLZOGetSize
//  Purpose:    returns structure size necessary for compression
//
//  Arguments:
//     method           LZO method to be used during compression
//     maxInputLen      maximum length of input buffer, which will be processed by Encode
//     pSize            pointer to size variable
//
//  Return:
//      ippStsBadArgErr          illegal method
//      ippStsNullPtrErr         NULL pointer detected
//      ippStsNoErr              no error
//
*/
IPPAPI(IppStatus, ippsEncodeLZOGetSize, (IppLZOMethod method, Ipp32u maxInputLen, Ipp32u *pSize))

/* /////////////////////////////////////////////////////////////////////////////
//  Name:       ippsEncodeLZO_8u
//  Purpose:    compresses specified input buffer
//
//  Arguments:
//      pSrc                   input data address
//      srcLen                 input data length
//      pDst                   output buffer address
//      pDstLen                pointer to resulting length variable, must contain output buffer length upon start
//      pLZOState              pointer to IppLZOState structure variable
//
//  Return:
//      ippStsNullPtrErr            one of the pointers is NULL
//      ippStsDstSizeLessExpected   output buffer is too short for compressed data
//      ippStsNoErr                 no error detected
//
*/
IPPAPI(IppStatus, ippsEncodeLZO_8u, (const Ipp8u *pSrc, Ipp32u srcLen, Ipp8u *pDst, Ipp32u *pDstLen, IppLZOState_8u *pLZOState ))

/* /////////////////////////////////////////////////////////////////////////////
//  Name:       ippsEncodeLZOInit
//  Purpose:    initializes IppLZOSate_8u structure
//
//  Arguments:
//      method                  LZO compression method desired
//      maxInputLen             maximum length of input buffer, which will be processed by Encode
//      pLZOState               pointer to IppLZOState structure variable
//
//  Return:
//      ippStsNullPtrErr            one of the pointers is NULL
//      ippStsBadArgErr             illegal method
//      ippStsNoErr                 no error detected
//
*/
IPPAPI(IppStatus, ippsEncodeLZOInit_8u, (IppLZOMethod method, Ipp32u maxInputLen, IppLZOState_8u *pLZOState))

/* /////////////////////////////////////////////////////////////////////////////
//  Name:       ippsDecodeLZO_8u
//      Purpose:    decompresses specified input buffer to output buffer, returns decompressed data length
//  Name:       ippsDecodeLZOSafe_8u
//      Purpose:    decompresses specified input buffer to output buffer with checking output buffer boundaries, returns decompressed data length
//
//  Arguments:
//      pSrc                  pointer to input buffer
//      srcLen                input data length
//      pDst                  pointer to output buffer
//      pDstLen               pointer to output data length variable. Initially contains output buffer length
//
//  Return:
//      ippStsNullPtrErr            one of the pointers is NULL
//      ippStsDstSizeLessExpected   output buffer is too short for compressed data
//      ippStsSrcSizeLessExpected   input buffer data is not complete, i.e. no EOF found
//      ippStsBrokenLzoStream       ippsDecodeLZOSafe_8u detected output buffer boundary violation
//      ippStsNoErr                 no error detected
//
*/
IPPAPI(IppStatus, ippsDecodeLZO_8u, (const Ipp8u *pSrc, Ipp32u srcLen, Ipp8u *pDst, Ipp32u *pDstLen))
IPPAPI(IppStatus, ippsDecodeLZOSafe_8u, (const Ipp8u *pSrc, Ipp32u srcLen, Ipp8u *pDst, Ipp32u *pDstLen))

/**************************************************
IPP RLE Definitions
***************************************************/
/* /////////////////////////////////////////////////////////////////////////////
//  Name:               ippsEncodeRLE_8u
//  Purpose:            Performs the RLE encoding
//
//  Parameters:
//    pSrc              Pointer to the source vector
//    pSrcLen           Pointer to the length of source vector on input,
//                      pointer to the size of remainder on output
//    pDst              Pointer to the destination vector
//    pDstLen           Pointer to the size of destination buffer on input,
//                      pointer to the resulting length of the destination vector
//                      on output.
//  Return:
//    ippStsNullPtrErr          One or several pointer(s) is NULL
//    ippStsSizeErr             Lengths of the source/destination vector are less
//                              or equal zero
//    ippStsDstSizeLessExpected The size of destination vector less expected
//    ippStsNoErr               No errors
//
*/
IPPAPI(IppStatus, ippsEncodeRLE_8u, ( Ipp8u** ppSrc, int* pSrcLen,
                                      Ipp8u* pDst, int* pDstLen ))

/* /////////////////////////////////////////////////////////////////////////////
//  Name:               ippsDecodeRLE_8u
//  Purpose:            Performs the RLE decoding
//
//  Parameters:
//    pSrc              Pointer to the source vector
//    pSrcLen           Pointer to the length of source vector on input,
//                      pointer to the size of remainder on output
//    pDst              Pointer to the destination vector
//    pDstLen           Pointer to the size of destination buffer on input,
//                      pointer to the resulting length of the destination vector
//                      on output.
//  Return:
//    ippStsNullPtrErr          One or several pointer(s) is NULL
//    ippStsSizeErr             Lengths of the source/destination vector are less
//                              or equal zero
//    ippStsDstSizeLessExpected The size of destination vector less expected
//    ippStsSrcDataErr          The source vector contains unsupported data
//    ippStsNoErr               No errors
//
*/
IPPAPI(IppStatus, ippsDecodeRLE_8u, ( Ipp8u** ppSrc, int* pSrcLen,
                                      Ipp8u* pDst, int* pDstLen ))

/**************************************************
IPP BZIP2 Definitions
***************************************************/

/* /////////////////////////////////////////////////////////////////////////////
//  Name:               ippsRLEGetSize_BZ2_8u
//  Purpose:            Calculates the size of internal state for bzip2-specific RLE.
//                      Specific function for bzip2 compatibility.
//
//  Parameters:
//    pRLEStateSize             Pointer to the size of internal state for bzip2-specific RLE
//
//  Return:
//    ippStsNullPtrErr          One or several pointer(s) is NULL
//    ippStsNoErr               No errors
//
*/

IPPAPI(IppStatus, ippsRLEGetSize_BZ2_8u,    ( int* pRLEStateSize ))

/* /////////////////////////////////////////////////////////////////////////////
//  Name:               ippsEncodeRLEInit_BZ2_8u
//  Purpose:            Initializes the elements of the bzip2-specific internal state for RLE.
//                      Specific function for bzip2 compatibility.
//
//  Parameters:
//    pRLEState         Pointer to internal state structure for bzip2 specific RLE
//
//  Return:
//    ippStsNullPtrErr          One or several pointer(s) is NULL
//    ippStsNoErr               No errors
//
*/

IPPAPI(IppStatus, ippsEncodeRLEInit_BZ2_8u,    ( IppRLEState_BZ2* pRLEState ))

/* /////////////////////////////////////////////////////////////////////////////
//  Name:               ippsEncodeRLE_BZ2_8u
//  Purpose:            Performs the RLE encoding with thresholding = 4.
//                      Specific function for bzip2 compatibility.
//
//  Parameters:
//    ppSrc             Double pointer to the source vector
//    pSrcLen           Pointer to the length of source vector
//    pDst              Pointer to the destination vector
//    pDstLen           Pointer to the size of destination buffer on input,
//                      pointer to the resulting length of the destination vector
//                      on output
//    pRLEState         Pointer to internal state structure for bzip2 specific RLE
//
//  Return:
//    ippStsNullPtrErr          One or several pointer(s) is NULL
//    ippStsSizeErr             Lengths of the source/destination vector are less
//                              or equal zero
//    ippStsDstSizeLessExpected The size of destination vector less expected
//    ippStsNoErr               No errors
//
*/

IPPAPI(IppStatus, ippsEncodeRLE_BZ2_8u,    ( Ipp8u** ppSrc, int* pSrcLen, Ipp8u* pDst, int* pDstLen, IppRLEState_BZ2* pRLEState ))

/* /////////////////////////////////////////////////////////////////////////////
//  Name:               ippsEncodeRLEFlush_BZ2_8u
//  Purpose:            Performs flushing the rest of data after RLE encoding with thresholding = 4.
//                      Specific function for bzip2 compatibility.
//
//  Parameters:
//    pDst              Pointer to the destination vector
//    pDstLen           Pointer to the size of destination buffer on input,
//                      pointer to the resulting length of the destination vector
//                      on output
//    pRLEState         Pointer to internal state structure for bzip2 specific RLE
//
//  Return:
//    ippStsNullPtrErr          One or several pointer(s) is NULL
//    ippStsSizeErr             Lengths of the source/destination vector are less
//                              or equal zero
//    ippStsNoErr               No errors
//
*/

IPPAPI(IppStatus, ippsEncodeRLEFlush_BZ2_8u,   ( Ipp8u* pDst, int* pDstLen, IppRLEState_BZ2* pRLEState ))

/* /////////////////////////////////////////////////////////////////////////////
//  Name:               ippsDecodeRLEStateInit_BZ2_8u
//  Purpose:            Initializes the elements of the bzip2-specific internal state for RLE.
//                      Specific function for bzip2 compatibility.
//
//  Parameters:
//    pRLEState         Pointer to internal state structure for bzip2 specific RLE
//
//  Return:
//    ippStsNullPtrErr          One or several pointer(s) is NULL
//    ippStsNoErr               No errors
//
*/

IPPAPI(IppStatus, ippsDecodeRLEStateInit_BZ2_8u, ( IppRLEState_BZ2* pRLEState ))

/* /////////////////////////////////////////////////////////////////////////////
//  Name:               ippsDecodeRLEState_BZ2_8u
//  Purpose:            Performs the RLE decoding with thresholding = 4.
//                      Specific function for bzip2 compatibility.
//
//  Parameters:
//    ppSrc             Double pointer to the source vector
//    pSrcLen           Pointer to the length of source vector
//    pDst              Double pointer to the destination vector
//    pDstLen           Pointer to the size of destination buffer on input,
//                      pointer to the resulting length of the destination vector
//                      on output
//    pRLEState         Pointer to internal state structure for bzip2 specific RLE
//
//  Return:
//    ippStsNullPtrErr          One or several pointer(s) is NULL
//    ippStsDstSizeLessExpected The size of destination vector less expected
//    ippStsNoErr               No errors
//
*/

IPPAPI(IppStatus, ippsDecodeRLEState_BZ2_8u, (Ipp8u** ppSrc, Ipp32u* pSrcLen, Ipp8u** ppDst, Ipp32u* pDstLen, IppRLEState_BZ2* pRLEState ))

/* /////////////////////////////////////////////////////////////////////////////
//  Name:               ippsDecodeRLEStateFlush_BZ2_8u
//  Purpose:            Performs flushing the rest of data after RLE decoding with thresholding = 4.
//                      Specific function for bzip2 compatibility.
//
//  Parameters:
//    pRLEState         Pointer to internal state structure for bzip2 specific RLE
//    ppDst             Double pointer to the destination vector
//    pDstLen           Pointer to the size of destination buffer on input,
//                      pointer to the resulting length of the destination vector
//                      on output
//
//  Return:
//    ippStsNullPtrErr          One or several pointer(s) is NULL
//    ippStsNoErr               No errors
//
*/

IPPAPI(IppStatus, ippsDecodeRLEStateFlush_BZ2_8u, (IppRLEState_BZ2* pRLEState, Ipp8u** ppDst, Ipp32u* pDstLen ))

/* /////////////////////////////////////////////////////////////////////////////
//  Name:               ippsRLEGetInUseTable_8u
//  Purpose:            Service function: gets the pointer to the inUse vector from internal state
//                      of type IppRLEState_BZ2. Specific function for bzip2 compatibility.
//
//  Parameters:
//    inUse             Pointer to the inUse vector
//    pRLEState         Pointer to internal state structure for bzip2 specific RLE
//
//  Return:
//    ippStsNullPtrErr          One or several pointer(s) is NULL
//    ippStsNoErr               No errors
//
*/

IPPAPI(IppStatus, ippsRLEGetInUseTable_8u,    ( Ipp8u inUse[256], IppRLEState_BZ2* pRLEState ))

/* /////////////////////////////////////////////////////////////////////////////
//  Name:               ippsEncodeZ1Z2_BZ2_8u16u
//  Purpose:            Performs the Z1Z2 encoding.
//                      Specific function for bzip2 compatibility.
//
//  Parameters:
//    ppSrc             Double pointer to the source vector
//    pSrcLen           Pointer to the length of source vector on input,
//                      pointer to the size of remainder on output
//    pDst              Pointer to the destination vector
//    pDstLen           Pointer to the size of destination buffer on input,
//                      pointer to the resulting length of the destination vector
//                      on output.
//    freqTable[258]    Table of frequencies collected for alphabet symbols.
//
//  Return:
//    ippStsNullPtrErr          One or several pointer(s) is NULL
//    ippStsSizeErr             Lengths of the source/destination vector are less
//                              or equal zero
//    ippStsDstSizeLessExpected The size of destination vector less expected
//    ippStsNoErr               No errors
//
*/

IPPAPI(IppStatus, ippsEncodeZ1Z2_BZ2_8u16u, ( Ipp8u** ppSrc, int* pSrcLen, Ipp16u* pDst, int* pDstLen, int freqTable[258] ))

/* /////////////////////////////////////////////////////////////////////////////
//  Name:               ippsDecodeZ1Z2_BZ2_16u8u
//  Purpose:            Performs the Z1Z2 decoding.
//                      Specific function for bzip2 compatibility.
//
//  Parameters:
//    ppSrc             Double pointer to the source vector
//    pSrcLen           Pointer to the length of source vector on input,
//                      pointer to the size of remainder on output
//    pDst              Pointer to the destination vector
//    pDstLen           Pointer to the size of destination buffer on input,
//                      pointer to the resulting length of the destination vector
//                      on output.
//
//  Return:
//    ippStsNullPtrErr          One or several pointer(s) is NULL
//    ippStsSizeErr             Lengths of the source/destination vector are less
//                              or equal zero
//    ippStsDstSizeLessExpected The size of destination vector less expected
//    ippStsNoErr               No errors
//
*/

IPPAPI(IppStatus, ippsDecodeZ1Z2_BZ2_16u8u, ( Ipp16u** ppSrc, int* pSrcLen, Ipp8u* pDst, int* pDstLen ))

/* /////////////////////////////////////////////////////////////////////////////
//  Name:               ippsReduceDictionary_8u_I
//  Purpose:            Performs the dictionary reducing.
//
//  Parameters:
//    inUse[256]        Table of 256 values of Ipp8u type.
//    pSrcDst           Pointer to the source/destination vector
//    srcDstLen         Length of source/destination vector.
//    pSizeDictionary   Pointer to the size of dictionary on input and to the size
//                      of reduced dictionary on output.
//
//  Return:
//    ippStsNullPtrErr          One or several pointer(s) is NULL
//    ippStsSizeErr             Lengths of the source/destination vector are less
//                              or equal zero
//    ippStsNoErr               No errors
//
*/

IPPAPI(IppStatus, ippsReduceDictionary_8u_I,   ( const Ipp8u inUse[256], Ipp8u* pSrcDst, int srcDstLen, int* pSizeDictionary ))

/* /////////////////////////////////////////////////////////////////////////////
//  Name:               ippsExpandDictionary_8u_I
//  Purpose:            Performs the dictionary expanding.
//
//  Parameters:
//    inUse[256]        Table of 256 values of Ipp8u type.
//    pSrcDst           Pointer to the source/destination vector
//    srcDstLen         Length of source/destination vector.
//    sizeDictionary    The size of reduced dictionary on input.
//
//  Return:
//    ippStsNullPtrErr          One or several pointer(s) is NULL
//    ippStsSizeErr             Lengths of the source/destination vector are less
//                              or equal zero
//    ippStsNoErr               No errors
//
*/

IPPAPI(IppStatus, ippsExpandDictionary_8u_I,   ( const Ipp8u inUse[256], Ipp8u* pSrcDst, int srcDstLen, int sizeDictionary ))

/* /////////////////////////////////////////////////////////////////////////////
//  Name:               ippsCRC32_BZ2_8u
//  Purpose:            Performs the CRC32 checksum calculation according to the direct algorithm, which is used in bzip2.
//
//  Parameters:
//    pSrc              Pointer to the source data vector
//    srcLen            The length of source vector
//    pCRC32            Pointer to the value of accumulated CRC32
//
//  Return:
//    ippStsNullPtrErr          One or several pointer(s) is NULL
//    ippStsSizeErr             Length of the source vector is less or equal zero
//    ippStsNoErr               No errors
//
*/

IPPAPI(IppStatus, ippsCRC32_BZ2_8u,    ( const Ipp8u* pSrc, int srcLen, Ipp32u* pCRC32 ))

/* /////////////////////////////////////////////////////////////////////////////
//  Name:               ippsEncodeHuffGetSize_BZ2_16u8u
//  Purpose:            Calculates the size of internal state for bzip2-specific Huffman coding.
//                      Specific function for bzip2 compatibility.
//
//  Parameters:
//    wndSize                          Size of the block to be processed
//    pEncodeHuffStateSize             Pointer to the size of internal state for bzip2-specific Huffman coding
//
//  Return:
//    ippStsNullPtrErr          One or several pointer(s) is NULL
//    ippStsSizeErr             Lengths of the source/destination vector are less
//                              or equal zero
//    ippStsNoErr               No errors
//
*/
IPPAPI(IppStatus, ippsEncodeHuffGetSize_BZ2_16u8u,   ( int wndSize, int* pEncodeHuffStateSize ))

/* /////////////////////////////////////////////////////////////////////////////
//  Name:               ippsEncodeHuffInit_BZ2_16u8u
//  Purpose:            Initializes the elements of the bzip2-specific internal state for Huffman coding.
//                      Specific function for bzip2 compatibility.
//
//  Parameters:
//    sizeDictionary     The size of the dictionary
//    freqTable          Table of frequencies of symbols
//    pSrc               Pointer to the source vector
//    srcLen             Length of the source vector
//    pEncodeHuffState   Pointer to internal state structure for bzip2 specific Huffman coding
//
//  Return:
//    ippStsNullPtrErr          One or several pointer(s) is NULL
//    ippStsSizeErr             Lengths of the source/destination vector are less
//                              or equal zero
//    ippStsNoErr               No errors
//
*/
IPPAPI(IppStatus, ippsEncodeHuffInit_BZ2_16u8u,      ( int sizeDictionary, const int freqTable[258], const Ipp16u* pSrc, int srcLen,
                                                       IppEncodeHuffState_BZ2* pEncodeHuffState ))

/* /////////////////////////////////////////////////////////////////////////////
//  Name:               ippsPackHuffContext_BZ2_16u8u
//  Purpose:            Performs the bzip2-specific encoding of Huffman context.
//                      Specific function for bzip2 compatibility.
//
//  Parameters:
//    pCode             Pointer to the bit buffer
//    pCodeLenBits      Number of valid bits in the bit buffer
//    pDst              Pointer to the destination vector
//    pDstLen           Pointer to the size of destination buffer on input,
//                      pointer to the resulting length of the destination vector
//                      on output
//    pEncodeHuffState  Pointer to internal state structure for bzip2 specific Huffman coding
//
//  Return:
//    ippStsNullPtrErr          One or several pointer(s) is NULL
//    ippStsSizeErr             Lengths of the source/destination vector are less
//                              or equal zero
//    ippStsDstSizeLessExpected The size of destination vector less expected
//    ippStsNoErr               No errors
//
*/
IPPAPI(IppStatus, ippsPackHuffContext_BZ2_16u8u,     ( Ipp32u* pCode, int* pCodeLenBits, Ipp8u* pDst, int* pDstLen,
                                                       IppEncodeHuffState_BZ2* pEncodeHuffState ))

/* /////////////////////////////////////////////////////////////////////////////
//  Name:               ippsEncodeHuff_BZ2_16u8u
//  Purpose:            Performs the bzip2-specific Huffman encoding.
//                      Specific function for bzip2 compatibility.
//
//  Parameters:
//    pCode             Pointer to the bit buffer
//    pCodeLenBits      Number of valid bits in the bit buffer
//    ppSrc             Double pointer to the source vector
//    pSrcLen           Pointer to the length of source vector
//    pDst              Pointer to the destination vector
//    pDstLen           Pointer to the size of destination buffer on input,
//                      pointer to the resulting length of the destination vector
//                      on output
//    pEncodeHuffState  Pointer to internal state structure for bzip2 specific Huffman coding
//
//  Return:
//    ippStsNullPtrErr          One or several pointer(s) is NULL
//    ippStsSizeErr             Lengths of the source/destination vector are less
//                              or equal zero
//    ippStsDstSizeLessExpected The size of destination vector less expected
//    ippStsNoErr               No errors
//
*/
IPPAPI(IppStatus, ippsEncodeHuff_BZ2_16u8u,          ( Ipp32u* pCode, int* pCodeLenBits, Ipp16u** ppSrc, int* pSrcLen,
                                                       Ipp8u* pDst, int* pDstLen, IppEncodeHuffState_BZ2* pEncodeHuffState ))

/* /////////////////////////////////////////////////////////////////////////////
//  Name:               ippsDecodeHuffGetSize_BZ2_8u16u
//  Purpose:            Calculates the size of internal state for bzip2-specific Huffman decoding.
//                      Specific function for bzip2 compatibility.
//
//  Parameters:
//    wndSize                    Size of the block to be processed
//    pDecodeHuffStateSize       Pointer to the size of internal state for bzip2-specific Huffman decoding
//
//  Return:
//    ippStsNullPtrErr          One or several pointer(s) is NULL
//    ippStsSizeErr             Lengths of the source/destination vector are less
//                              or equal zero
//    ippStsNoErr               No errors
//
*/
IPPAPI(IppStatus, ippsDecodeHuffGetSize_BZ2_8u16u,   ( int wndSize, int* pDecodeHuffStateSize ))

/* /////////////////////////////////////////////////////////////////////////////
//  Name:               ippsDecodeHuffInit_BZ2_8u16u
//  Purpose:            Initializes the elements of the bzip2-specific internal state for Huffman decoding.
//                      Specific function for bzip2 compatibility.
//
//  Parameters:
//    sizeDictionary           The size of the dictionary
//    pDecodeHuffState         Pointer to internal state structure for bzip2 specific Huffman decoding
//
//  Return:
//    ippStsNullPtrErr          One or several pointer(s) is NULL
//    ippStsSizeErr             Lengths of the source/destination vector are less
//                              or equal zero
//    ippStsNoErr               No errors
//
*/
IPPAPI(IppStatus, ippsDecodeHuffInit_BZ2_8u16u,      ( int sizeDictionary, IppDecodeHuffState_BZ2* pDecodeHuffState ))

/* /////////////////////////////////////////////////////////////////////////////
//  Name:               ippsUnpackHuffContext_BZ2_8u16u
//  Purpose:            Performs the bzip2-specific decoding of Huffman context.
//                      Specific function for bzip2 compatibility.
//
//  Parameters:
//    pCode                   Pointer to the bit buffer
//    pCodeLenBits            Number of valid bits in the bit buffer
//    pSrc                    Pointer to the destination vector
//    pSrcLen                 Pointer to the size of destination buffer on input,
//                            pointer to the resulting length of the destination vector
//                            on output
//    pDecodeHuffState        Pointer to internal state structure for bzip2 specific Huffman decoding.
//
//  Return:
//    ippStsNullPtrErr          One or several pointer(s) is NULL
//    ippStsSizeErr             Lengths of the source/destination vector are less
//                              or equal zero
//    ippStsDstSizeLessExpected The size of destination vector less expected
//    ippStsNoErr               No errors
//
*/
IPPAPI(IppStatus, ippsUnpackHuffContext_BZ2_8u16u,   ( Ipp32u* pCode, int* pCodeLenBits, Ipp8u** ppSrc, int* pSrcLen,
                                                       IppDecodeHuffState_BZ2* pDecodeHuffState ))

/* /////////////////////////////////////////////////////////////////////////////
//  Name:               ippsDecodeHuff_BZ2_8u16u
//  Purpose:            Performs the bzip2-specific Huffman decoding.
//                      Specific function for bzip2 compatibility.
//
//  Parameters:
//    pCode                   Pointer to the bit buffer
//    pCodeLenBits            Number of valid bits in the bit buffer
//    ppSrc                   Double pointer to the source vector
//    pSrcLen                 Pointer to the length of source vector
//    pDst                    Pointer to the destination vector
//    pDstLen                 Pointer to the size of destination buffer on input,
//                            pointer to the resulting length of the destination vector
//                            on output
//    pDecodeHuffState        Pointer to internal state structure for bzip2 specific Huffman decoding.
//
//  Return:
//    ippStsNullPtrErr          One or several pointer(s) is NULL
//    ippStsSizeErr             Lengths of the source/destination vector are less
//                              or equal zero
//    ippStsDstSizeLessExpected The size of destination vector less expected
//    ippStsNoErr               No errors
//
*/
IPPAPI(IppStatus, ippsDecodeHuff_BZ2_8u16u,          ( Ipp32u* pCode, int* pCodeLenBits, Ipp8u** ppSrc, int* pSrcLen,
                                                       Ipp16u* pDst, int* pDstLen, IppDecodeHuffState_BZ2* pDecodeHuffState ))

/* /////////////////////////////////////////////////////////////////////////////
// Name:                ippsDecodeBlockGetSize_BZ2_8u
// Purpose:             Computes the size of necessary memory (in bytes) for
//                      additional buffer for the bzip2-specific decoding.
//                      Specific function for bzip2 compatibility.
//
// Parameters:
//    blockSize         Block size for the bzip2-specific decoding
//    pBuffSize         Pointer to the computed size of buffer
//
// Return:
//    ippStsNullPtrErr  Pointer is NULL
//    ippStsNoErr       No errors
//
*/
IPPAPI(IppStatus, ippsDecodeBlockGetSize_BZ2_8u, ( int blockSize, int* pBuffSize ))

/* /////////////////////////////////////////////////////////////////////////////
// Name:                ippsDecodeBlockGetSize_BZ2_8u
//  Purpose:            Performs the bzip2-specific block decoding.
//                      Specific function for bzip2 compatibility.
//
// Parameters:
//    pSrc              Pointer to the source vector
//    pSrcLen           Pointer to the length of source vector
//    pDst              Pointer to the destination vector
//    pDstLen           Pointer to the size of destination buffer on input,
//                      pointer to the resulting length of the destination buffer
//                      on output
//    index             Index of first position for the inverse BWT transform
//    dictSize          The size of reduced dictionary
//    inUse             Table of 256 values of Ipp8u type
//    pBuff             Pointer to the additional buffer
//
// Return:
//    ippStsNullPtrErr  One or several pointer(s) is NULL
//    ippStsSizeErr     Length of source/destination vectors is less or
//                      equal zero or index greater or equal srcLen
//    ippStsNoErr       No errors
//
*/
IPPAPI(IppStatus, ippsDecodeBlock_BZ2_16u8u, ( const Ipp16u* pSrc, int srcLen, Ipp8u* pDst, int* pDstLen,
                                               int index, int dictSize, const Ipp8u inUse[256], Ipp8u* pBuff ))

/*BWT service functionality for BZIP2*/

/* /////////////////////////////////////////////////////////////////////////////
// Name:                ippsBWTFwdGetBufSize_SelectSort_8u
// Purpose:             Computes the size of necessary memory (in bytes) for
//                      additional buffer for the forward BWT transform
//
// Parameters:
//    wndSize           Window size for the BWT transform
//    pBWTFwdBufSize    Pointer to the computed size of buffer
//    sortAlgorithmHint Strategy hint for Sort algorithm selection
//
// Return:
//    ippStsNullPtrErr  Pointer is NULL
//    ippStsNoErr       No errors
//
*/
IPPAPI(IppStatus, ippsBWTFwdGetBufSize_SelectSort_8u, (Ipp32u wndSize, Ipp32u* pBWTFwdBufSize,
                                                       IppBWTSortAlgorithmHint sortAlgorithmHint ))

/* /////////////////////////////////////////////////////////////////////////////
//  Name:               ippsBWTFwd_SelectSort_8u
//  Purpose:            Performs the forward BWT transform
//
//  Parameters:
//    pSrc              Pointer to the source vector
//    pDst              Pointer to the destination vector
//    len               Length of source/destination vectors
//    index             Pointer to the index of first position for
//                      the inverse BWT transform
//    pBWTFwdBuf        Pointer to the additional buffer
//    sortAlgorithmHint Strategy hint for Sort algorithm selection
//
//  Return:
//    ippStsNullPtrErr  One or several pointer(s) is NULL
//    ippStsSizeErr     Length of source/destination vectors is less or equal zero
//    ippStsNoErr       No errors
//
*/
IPPAPI(IppStatus, ippsBWTFwd_SelectSort_8u, ( const Ipp8u* pSrc, Ipp8u* pDst, Ipp32u len, Ipp32u* index, Ipp8u* pBWTFwdBuf,
                                              IppBWTSortAlgorithmHint sortAlgorithmHint ))

/* Move To Front service functionality for BZIP2 */

/* /////////////////////////////////////////////////////////////////////////////
// Name:                ippsMTFInit_8u
// Purpose:             Initializes parameters for the MTF transform
//
// Parameters:
//    pMTFState         Pointer to the structure containing parameters for
//                      the MTF transform
//
// Return:
//    ippStsNullPtrErr  Pointer to structure is NULL
//    ippStsNoErr       No errors
//
*/
IPPAPI(IppStatus, ippsMTFInit_8u, ( IppMTFState_8u* pMTFState ))

/* /////////////////////////////////////////////////////////////////////////////
// Name:                ippsMTFGetSize_8u
// Purpose:             Computes the size of necessary memory (in bytes) for
//                      structure of the MTF transform
//
// Parameters:
//    pMTFStateSize     Pointer to the computed size of structure
//
// Return:
//    ippStsNullPtrErr  Pointer is NULL
//    ippStsNoErr       No errors
//
*/
IPPAPI(IppStatus, ippsMTFGetSize_8u, ( int* pMTFStateSize ))

/* /////////////////////////////////////////////////////////////////////////////
//  Name:               ippsMTFFwd_8u
//  Purpose:            Performs the forward MTF transform
//
//  Parameters:
//    pSrc              Pointer to the source vector
//    pDst              Pointer to the destination vector
//    len               Length of source/destination vectors
//    pMTFState         Pointer to the structure containing parameters for
//                      the MTF transform
//  Return:
//    ippStsNullPtrErr  One or several pointer(s) is NULL
//    ippStsSizeErr     Length of the source vector is less or equal zero
//    ippStsNoErr       No errors
//
*/
IPPAPI(IppStatus, ippsMTFFwd_8u, ( const Ipp8u* pSrc, Ipp8u* pDst, int len,
                                   IppMTFState_8u* pMTFState ))

/*service functionality*/
IPPAPI ( IppStatus, ippsSet_32s,( Ipp32s val, Ipp32s* pDst, int len ))
IPPAPI ( IppStatus, ippsMove_8u,( const Ipp8u* pSrc, Ipp8u* pDst, int len ))
IPPAPI(IppStatus, ippsCopy_1u,
      ( const Ipp8u* pSrc, int srcBitOffset, Ipp8u* pDst, int dstBitOffset, int len ))
IPPAPI(IppStatus, ippsCopy_8u,( const Ipp8u* pSrc, Ipp8u* pDst, int len ))
IPPAPI ( IppStatus, ippsZero_8u,( Ipp8u* pDst, int len ))

#ifdef __cplusplus
}
#endif

#endif /* __IPPFPK_H__ */
