/*******************************************************************************
* Copyright 2014-2016 Intel Corporation
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

/*-------------------------------------------------------------*/
/*--- Public header file for the library.                   ---*/
/*---                                               bzlib.h ---*/
/*-------------------------------------------------------------*/

/* ------------------------------------------------------------------
   This file is part of bzip2/libbzip2, a program and library for
   lossless, block-sorting data compression.

   bzip2/libbzip2 version 1.0.4 of 20 December 2006
   Copyright (C) 1996-2006 Julian Seward <jseward@bzip.org>

   Please read the WARNING, DISCLAIMER and PATENTS sections in the
   README file.

   This program is released under the terms of the license contained
   in the file LICENSE.
   ------------------------------------------------------------------ */
/*
 * This source code file was modified with Intel(R) Integrated Performance Primitives content
 */
#ifndef _BZLIB_H
#define _BZLIB_H

#define FPK_PREFIX

#include "ipp.h"
/*--
   If you *really* need the "ipp_" prefix for all library functions,
   compile with -DIPP_PREFIX. The "standard" libbzip2 should be
   compiled without it.
--*/
#ifdef IPP_PREFIX
#  define BZ2_bzCompressInit          ipp_BZ2_bzCompressInit
#  define BZ2_bzCompress              ipp_BZ2_bzCompress
#  define BZ2_bzCompressEnd           ipp_BZ2_bzCompressEnd
#  define BZ2_bzDecompressInit        ipp_BZ2_bzDecompressInit
#  define BZ2_bzDecompress            ipp_BZ2_bzDecompress
#  define BZ2_bzDecompressEnd         ipp_BZ2_bzDecompressEnd
#  define BZ2_bzReadOpen              ipp_BZ2_bzReadOpen
#  define BZ2_bzReadClose             ipp_BZ2_bzReadClose
#  define BZ2_bzReadGetUnused         ipp_BZ2_bzReadGetUnused
#  define BZ2_bzRead                  ipp_BZ2_bzRead
#  define BZ2_bzWriteOpen             ipp_BZ2_bzWriteOpen
#  define BZ2_bzWrite                 ipp_BZ2_bzWrite
#  define BZ2_bzWriteClose            ipp_BZ2_bzWriteClose
#  define BZ2_bzWriteClose64          ipp_BZ2_bzWriteClose64
#  define BZ2_bzBuffToBuffCompress    ipp_BZ2_bzBuffToBuffCompress
#  define BZ2_bzBuffToBuffDecompress  ipp_BZ2_bzBuffToBuffDecompress
#  define BZ2_bzlibVersion            ipp_BZ2_bzlibVersion
#  define BZ2_bzopen                  ipp_BZ2_bzopen
#  define BZ2_bzdopen                 ipp_BZ2_bzdopen
#  define BZ2_bzread                  ipp_BZ2_bzread
#  define BZ2_bzwrite                 ipp_BZ2_bzwrite
#  define BZ2_bzflush                 ipp_BZ2_bzflush
#  define BZ2_bzclose                 ipp_BZ2_bzclose
#  define BZ2_bzerror                 ipp_BZ2_bzerror

#  define BZ2_blockSort               ipp_BZ2_blockSort
#  define BZ2_compressBlock           ipp_BZ2_compressBlock
#  define BZ2_bsInitWrite             ipp_BZ2_bsInitWrite
#  define BZ2_hbAssignCodes           ipp_BZ2_hbAssignCodes
#  define BZ2_hbMakeCodeLengths       ipp_BZ2_hbMakeCodeLengths
#  define BZ2_indexIntoF              ipp_BZ2_indexIntoF
#  define BZ2_decompress              ipp_BZ2_decompress
#  define BZ2_hbCreateDecodeTables    ipp_BZ2_hbCreateDecodeTables
#  define BZ2_bz__AssertH__fail       ipp_BZ2_bz__AssertH__fail
#  define BZ2_rNums                   ipp_BZ2_rNums
#  define BZ2_crc32Table              ipp_BZ2_crc32Table
#endif

#ifdef FPK_PREFIX
#  define BZ2_bzCompressInit1          fpk_BZ2_bzCompressInit_std
#  define BZ2_bzCompress1              fpk_BZ2_bzCompress_std
#  define BZ2_bzCompressEnd1           fpk_BZ2_bzCompressEnd_std
#  define BZ2_bzDecompressInit1        fpk_BZ2_bzDecompressInit_std
#  define BZ2_bzDecompress1            fpk_BZ2_bzDecompress_std
#  define BZ2_bzDecompressEnd1         fpk_BZ2_bzDecompressEnd_std

#  define BZ2_bzCompressInit          fpk_BZ2_bzCompressInit
#  define BZ2_bzCompress              fpk_BZ2_bzCompress
#  define BZ2_bzCompressEnd           fpk_BZ2_bzCompressEnd
#  define BZ2_bzDecompressInit        fpk_BZ2_bzDecompressInit
#  define BZ2_bzDecompress            fpk_BZ2_bzDecompress
#  define BZ2_bzDecompressEnd         fpk_BZ2_bzDecompressEnd
#  define BZ2_bzReadOpen              fpk_BZ2_bzReadOpen
#  define BZ2_bzReadClose             fpk_BZ2_bzReadClose
#  define BZ2_bzReadGetUnused         fpk_BZ2_bzReadGetUnused
#  define BZ2_bzRead                  fpk_BZ2_bzRead
#  define BZ2_bzWriteOpen             fpk_BZ2_bzWriteOpen
#  define BZ2_bzWrite                 fpk_BZ2_bzWrite
#  define BZ2_bzWriteClose            fpk_BZ2_bzWriteClose
#  define BZ2_bzWriteClose64          fpk_BZ2_bzWriteClose64
#  define BZ2_bzBuffToBuffCompress    fpk_BZ2_bzBuffToBuffCompress
#  define BZ2_bzBuffToBuffDecompress  fpk_BZ2_bzBuffToBuffDecompress
#  define BZ2_bzlibVersion            fpk_BZ2_bzlibVersion
#  define BZ2_bzopen                  fpk_BZ2_bzopen
#  define BZ2_bzdopen                 fpk_BZ2_bzdopen
#  define BZ2_bzread                  fpk_BZ2_bzread
#  define BZ2_bzwrite                 fpk_BZ2_bzwrite
#  define BZ2_bzflush                 fpk_BZ2_bzflush
#  define BZ2_bzclose                 fpk_BZ2_bzclose
#  define BZ2_bzerror                 fpk_BZ2_bzerror

#  define BZ2_blockSort               fpk_BZ2_blockSort
#  define BZ2_compressBlock           fpk_BZ2_compressBlock
#  define BZ2_bsInitWrite             fpk_BZ2_bsInitWrite
#  define BZ2_hbAssignCodes           fpk_BZ2_hbAssignCodes
#  define BZ2_hbMakeCodeLengths       fpk_BZ2_hbMakeCodeLengths
#  define BZ2_indexIntoF              fpk_BZ2_indexIntoF
#  define BZ2_decompress              fpk_BZ2_decompress
#  define BZ2_hbCreateDecodeTables    fpk_BZ2_hbCreateDecodeTables
#  define BZ2_bz__AssertH__fail       fpk_BZ2_bz__AssertH__fail
#  define BZ2_rNums                   fpk_BZ2_rNums
#  define BZ2_crc32Table              fpk_BZ2_crc32Table
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define BZ_RUN               0
#define BZ_FLUSH             1
#define BZ_FINISH            2

#define BZ_OK                0
#define BZ_RUN_OK            1
#define BZ_FLUSH_OK          2
#define BZ_FINISH_OK         3
#define BZ_STREAM_END        4
#define BZ_SEQUENCE_ERROR    (-1)
#define BZ_PARAM_ERROR       (-2)
#define BZ_MEM_ERROR         (-3)
#define BZ_DATA_ERROR        (-4)
#define BZ_DATA_ERROR_MAGIC  (-5)
#define BZ_IO_ERROR          (-6)
#define BZ_UNEXPECTED_EOF    (-7)
#define BZ_OUTBUFF_FULL      (-8)
#define BZ_CONFIG_ERROR      (-9)

typedef
   struct {
      char *next_in;
      unsigned int avail_in;
      unsigned int total_in_lo32;
      unsigned int total_in_hi32;

      char *next_out;
      unsigned int avail_out;
      unsigned int total_out_lo32;
      unsigned int total_out_hi32;

      void *state;

      void *(*bzalloc)(void *,int,int);
      void (*bzfree)(void *,void *);
      void *opaque;
   }
   bz_stream;


#ifndef BZ_IMPORT
#define BZ_EXPORT
#endif

#ifndef BZ_NO_STDIO
/* Need a definitition for FILE */
#include <stdio.h>
#endif

//#ifdef _WIN32
//#   include <windows.h>
//#   ifdef small
      /* windows.h define small to char */
//#      undef small
//#   endif
//#   ifdef BZ_EXPORT
//#   define BZ_API(func) WINAPI func
//#   define BZ_EXTERN extern
//#   else
   /* import windows dll dynamically */
//#   define BZ_API(func) (WINAPI * func)
//#   define BZ_EXTERN
//#   endif
//#else
//#   define BZ_API(func) func
//#   define BZ_EXTERN extern
//#endif

#undef _DLL

#if defined(WIN32) || defined(_WIN32)
//# include <windows.h>
# ifdef small
    /* windows.h define small to char */
#    undef small
# endif
# if defined(_DLL) && !defined(_LIB)
#   if !defined(BZ_EXPORT)
#     pragma message( "BZIP compiling as DLL import" )
#     define BZ_API(func) func
#     define BZ_EXTERN __declspec(dllimport)
#   else
#     pragma message( "BZIP compiling as DLL export" )
#     define BZ_API(func) func
#     define BZ_EXTERN extern __declspec(dllexport)
#   endif
# else
#   pragma message( "BZIP compiling as library" )
#   define BZ_API(func) func
#   define BZ_EXTERN extern
# endif
#else
#   define BZ_API(func) func
#   define BZ_EXTERN extern
#endif

/*-- Core (low-level) library functions --*/

BZ_EXTERN int BZ_API(BZ2_bzCompressInit) (
      bz_stream* strm,
      int        blockSize100k,
      int        verbosity,
      int        workFactor
   );

BZ_EXTERN int BZ_API(BZ2_bzCompress) (
      bz_stream* strm,
      int action
   );

BZ_EXTERN int BZ_API(BZ2_bzCompressEnd) (
      bz_stream* strm
   );

BZ_EXTERN int BZ_API(BZ2_bzDecompressInit) (
      bz_stream *strm,
      int       verbosity,
      int       small
   );

BZ_EXTERN int BZ_API(BZ2_bzDecompress) (
      bz_stream* strm
   );

BZ_EXTERN int BZ_API(BZ2_bzDecompressEnd) (
      bz_stream *strm
   );

/* temporary workaround */
BZ_EXTERN int BZ_API(BZ2_bzCompressInit1) (
      bz_stream* strm,
      int        blockSize100k,
      int        verbosity,
      int        workFactor
   );

BZ_EXTERN int BZ_API(BZ2_bzCompress1) (
      bz_stream* strm,
      int action
   );

BZ_EXTERN int BZ_API(BZ2_bzCompressEnd1) (
      bz_stream* strm
   );

BZ_EXTERN int BZ_API(BZ2_bzDecompressInit1) (
      bz_stream *strm,
      int       verbosity,
      int       small
   );

BZ_EXTERN int BZ_API(BZ2_bzDecompress1) (
      bz_stream* strm
   );

BZ_EXTERN int BZ_API(BZ2_bzDecompressEnd1) (
      bz_stream *strm
   );
/* end of temporary workaround */

/*-- High(er) level library functions --*/

#ifndef BZ_NO_STDIO
#define BZ_MAX_UNUSED 5000

typedef void BZFILE;

BZ_EXTERN BZFILE* BZ_API(BZ2_bzReadOpen) (
      int*  bzerror,
      FILE* f,
      int   verbosity,
      int   small,
      void* unused,
      int   nUnused
   );

BZ_EXTERN void BZ_API(BZ2_bzReadClose) (
      int*    bzerror,
      BZFILE* b
   );

BZ_EXTERN void BZ_API(BZ2_bzReadGetUnused) (
      int*    bzerror,
      BZFILE* b,
      void**  unused,
      int*    nUnused
   );

BZ_EXTERN int BZ_API(BZ2_bzRead) (
      int*    bzerror,
      BZFILE* b,
      void*   buf,
      int     len
   );

BZ_EXTERN BZFILE* BZ_API(BZ2_bzWriteOpen) (
      int*  bzerror,
      FILE* f,
      int   blockSize100k,
      int   verbosity,
      int   workFactor
   );

BZ_EXTERN void BZ_API(BZ2_bzWrite) (
      int*    bzerror,
      BZFILE* b,
      void*   buf,
      int     len
   );

BZ_EXTERN void BZ_API(BZ2_bzWriteClose) (
      int*          bzerror,
      BZFILE*       b,
      int           abandon,
      unsigned int* nbytes_in,
      unsigned int* nbytes_out
   );

BZ_EXTERN void BZ_API(BZ2_bzWriteClose64) (
      int*          bzerror,
      BZFILE*       b,
      int           abandon,
      unsigned int* nbytes_in_lo32,
      unsigned int* nbytes_in_hi32,
      unsigned int* nbytes_out_lo32,
      unsigned int* nbytes_out_hi32
   );
#endif


/*-- Utility functions --*/

BZ_EXTERN int BZ_API(BZ2_bzBuffToBuffCompress) (
      char*         dest,
      unsigned int* destLen,
      char*         source,
      unsigned int  sourceLen,
      int           blockSize100k,
      int           verbosity,
      int           workFactor
   );

BZ_EXTERN int BZ_API(BZ2_bzBuffToBuffDecompress) (
      char*         dest,
      unsigned int* destLen,
      char*         source,
      unsigned int  sourceLen,
      int           small,
      int           verbosity
   );


/*--
   Code contributed by Yoshioka Tsuneo (tsuneo@rr.iij4u.or.jp)
   to support better zlib compatibility.
   This code is not _officially_ part of libbzip2 (yet);
   I haven't tested it, documented it, or considered the
   threading-safeness of it.
   If this code breaks, please contact both Yoshioka and me.
--*/

BZ_EXTERN const char * BZ_API(BZ2_bzlibVersion) (
      void
   );

#ifndef BZ_NO_STDIO
BZ_EXTERN BZFILE * BZ_API(BZ2_bzopen) (
      const char *path,
      const char *mode
   );

BZ_EXTERN BZFILE * BZ_API(BZ2_bzdopen) (
      int        fd,
      const char *mode
   );

BZ_EXTERN int BZ_API(BZ2_bzread) (
      BZFILE* b,
      void* buf,
      int len
   );

BZ_EXTERN int BZ_API(BZ2_bzwrite) (
      BZFILE* b,
      void*   buf,
      int     len
   );

BZ_EXTERN int BZ_API(BZ2_bzflush) (
      BZFILE* b
   );

BZ_EXTERN void BZ_API(BZ2_bzclose) (
      BZFILE* b
   );

BZ_EXTERN const char * BZ_API(BZ2_bzerror) (
      BZFILE *b,
      int    *errnum
   );
#endif

#ifdef __cplusplus
}
#endif

#endif

/*-------------------------------------------------------------*/
/*--- end                                           bzlib.h ---*/
/*-------------------------------------------------------------*/
