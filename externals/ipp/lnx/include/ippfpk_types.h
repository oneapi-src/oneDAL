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

#if !defined( __IPPFPK_TYPES_H__ )
#define __IPPFPK_TYPES_H__

#ifdef __cplusplus
extern "C" {
#endif

typedef struct IppInflateState {
  const Ipp8u* pWindow;          /* pointer to the sliding window
                                    (the dictionary for the LZ77 algorithm) */
  unsigned int winSize;          /* size of the sliding window */
  unsigned int tableType;        /* type of Huffman code tables
                                    (for example, 0 - tables for Fixed Huffman codes) */
  unsigned int tableBufferSize;  /* (ENOUGH = 2048) * (sizeof(code) = 4) -
                                    sizeof(IppInflateState) */
} IppInflateState;

typedef enum { /* this type is used as a translator of the inflate_mode type from zlib */
  ippTYPE,
  ippLEN,
  ippLENEXT
} IppInflateMode;

typedef struct {
  Ipp16u freq;
  Ipp16u code;
} IppDeflateFreqTable;

typedef struct {
  Ipp16u code;
  Ipp16u len;
} IppDeflateHuffCode;

typedef enum {
   IppLZ77NoFlush,
   IppLZ77SyncFlush,
   IppLZ77FullFlush,
   IppLZ77FinishFlush
} IppLZ77Flush;

typedef enum {
    IppLZO1XST,      /* Single-threaded, generic LZO-compatible*/
    IppLZO1XMT      /* Multi-threaded */
} IppLZOMethod ;
struct LZOState_8u;
typedef struct LZOState_8u IppLZOState_8u;

struct RLEState_BZ2;
typedef struct RLEState_BZ2 IppRLEState_BZ2;

struct EncodeHuffState_BZ2;
typedef struct EncodeHuffState_BZ2 IppEncodeHuffState_BZ2;

struct DecodeHuffState_BZ2;
typedef struct DecodeHuffState_BZ2 IppDecodeHuffState_BZ2;

typedef enum {
    ippBWTItohTanakaLimSort,
    ippBWTItohTanakaUnlimSort,
    ippBWTSuffixSort,
    ippBWTAutoSort
} IppBWTSortAlgorithmHint;

struct MTFState_8u;
typedef struct MTFState_8u IppMTFState_8u;

#ifdef __cplusplus
}
#endif

#endif /* __IPPFPK_TYPES_H__ */
