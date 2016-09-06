/*******************************************************************************
* Copyright 1999-2016 Intel Corporation
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
//             Intel(R) Integrated Performance Primitives
//             Common Types and Macro Definitions
//
*/

#ifndef __IPPDEFS_H__
#define __IPPDEFS_H__

#ifdef __cplusplus
extern "C" {
#endif


#if defined( _IPP_PARALLEL_STATIC ) || defined( _IPP_PARALLEL_DYNAMIC )
  #pragma message("Threaded versions of IPP libraries are deprecated and will be removed in one of the future IPP releases.")
#endif

#if defined (_WIN64)
#define _INTEL_PLATFORM "intel64/"
#elif defined (_WIN32)
#define _INTEL_PLATFORM "ia32/"
#endif

#if defined( _WIN32 ) || defined ( _WIN64 )
  #define __STDCALL  __stdcall
  #define __CDECL    __cdecl
  #define __INT64    __int64
  #define __UINT64    unsigned __int64
#else
  #define __STDCALL
  #define __CDECL
  #define __INT64    long long
  #define __UINT64    unsigned long long
#endif


#if !defined( IPPAPI )

  #if defined( IPP_W32DLL ) && (defined( _WIN32 ) || defined( _WIN64 ))
    #if defined( _MSC_VER ) || defined( __ICL )
      #define IPPAPI( type,name,arg ) \
                     __declspec(dllimport)   type __STDCALL name arg;
    #else
      #define IPPAPI( type,name,arg )        type __STDCALL name arg;
    #endif
  #else
    #define   IPPAPI( type,name,arg )        type __STDCALL name arg;
  #endif

#endif

#if (defined( __ICL ) || defined( __ECL ) || defined(_MSC_VER)) && !defined( _PCS ) && !defined( _PCS_GENSTUBS )
  #if( __INTEL_COMPILER >= 1100 ) /* icl 11.0 supports additional comment */
    #if( _MSC_VER >= 1400 )
      #define IPP_DEPRECATED( comment ) __declspec( deprecated ( comment ))
    #else
      #pragma message ("your icl version supports additional comment for deprecated functions but it can't be displayed")
      #pragma message ("because internal _MSC_VER macro variable setting requires compatibility with MSVC7.1")
      #pragma message ("use -Qvc8 switch for icl command line to see these additional comments")
      #define IPP_DEPRECATED( comment ) __declspec( deprecated )
    #endif
  #elif( _MSC_FULL_VER >= 140050727 )&&( !defined( __INTEL_COMPILER )) /* VS2005 supports additional comment */
    #define IPP_DEPRECATED( comment ) __declspec( deprecated ( comment ))
  #elif( _MSC_VER <= 1200 )&&( !defined( __INTEL_COMPILER )) /* VS 6 doesn't support deprecation */
    #define IPP_DEPRECATED( comment )
  #else
    #define IPP_DEPRECATED( comment ) __declspec( deprecated )
  #endif
#elif (defined(__ICC) || defined(__ECC) || defined( __GNUC__ )) && !defined( _PCS ) && !defined( _PCS_GENSTUBS )
  #if defined( __GNUC__ )
    #if __GNUC__ >= 4 && __GNUC_MINOR__ >= 5
      #define IPP_DEPRECATED( message ) __attribute__(( deprecated( message )))
    #else
      #define IPP_DEPRECATED( message ) __attribute__(( deprecated ))
    #endif
  #else
    #define IPP_DEPRECATED( comment ) __attribute__(( deprecated ))
  #endif
#else
  #define IPP_DEPRECATED( comment )
#endif

#if (defined( __ICL ) || defined( __ECL ) || defined(_MSC_VER))
  #if !defined( _IPP_NO_DEFAULT_LIB )
    #if  (( defined( _IPP_PARALLEL_DYNAMIC ) && !defined( _IPP_PARALLEL_STATIC ) && \
          !defined( _IPP_SEQUENTIAL_DYNAMIC ) && !defined( _IPP_SEQUENTIAL_STATIC )) || \
          (!defined( _IPP_PARALLEL_DYNAMIC ) &&  defined( _IPP_PARALLEL_STATIC ) && \
          !defined( _IPP_SEQUENTIAL_DYNAMIC ) && !defined( _IPP_SEQUENTIAL_STATIC )) || \
          (!defined( _IPP_PARALLEL_DYNAMIC ) && !defined( _IPP_PARALLEL_STATIC ) && \
           defined( _IPP_SEQUENTIAL_DYNAMIC ) && !defined( _IPP_SEQUENTIAL_STATIC )) || \
          (!defined( _IPP_PARALLEL_DYNAMIC ) && !defined( _IPP_PARALLEL_STATIC ) && \
          !defined( _IPP_SEQUENTIAL_DYNAMIC ) &&  defined( _IPP_SEQUENTIAL_STATIC )))
    #elif (!defined( _IPP_PARALLEL_DYNAMIC ) && !defined( _IPP_PARALLEL_STATIC ) && \
          !defined( _IPP_SEQUENTIAL_DYNAMIC ) && !defined( _IPP_SEQUENTIAL_STATIC ))
      #define _IPP_NO_DEFAULT_LIB
    #else
      #error Illegal combination of _IPP_PARALLEL_DYNAMIC/_IPP_PARALLEL_STATIC/_IPP_SEQUENTIAL_DYNAMIC/_IPP_SEQUENTIAL_STATIC, \
             only one definition can be defined
    #endif
  #endif
#else
  #define _IPP_NO_DEFAULT_LIB
  #if (defined( _IPP_PARALLEL_DYNAMIC ) || defined( _IPP_PARALLEL_STATIC ) || \
       defined(_IPP_SEQUENTIAL_DYNAMIC) || defined(_IPP_SEQUENTIAL_STATIC))
#pragma message ("defines _IPP_PARALLEL_DYNAMIC/STATIC/_IPP_SEQUENTIAL_DYNAMIC/STATIC do not have any effect in current configuration")
  #endif
#endif

#if !defined( _IPP_NO_DEFAULT_LIB )
  #if defined( _IPP_PARALLEL_STATIC )
    #pragma comment( lib, "libircmt" )
    #pragma comment( lib, "libmmt" )
    #pragma comment( lib, "svml_dispmt" )
    #pragma comment( lib, "libiomp5md" )
  #endif
#endif



#define IPP_PI    ( 3.14159265358979323846 )  /* ANSI C does not support M_PI */
#define IPP_2PI   ( 6.28318530717958647692 )  /* 2*pi                         */
#define IPP_PI2   ( 1.57079632679489661923 )  /* pi/2                         */
#define IPP_PI4   ( 0.78539816339744830961 )  /* pi/4                         */
#define IPP_PI180 ( 0.01745329251994329577 )  /* pi/180                       */
#define IPP_RPI   ( 0.31830988618379067154 )  /* 1/pi                         */
#define IPP_SQRT2 ( 1.41421356237309504880 )  /* sqrt(2)                      */
#define IPP_SQRT3 ( 1.73205080756887729353 )  /* sqrt(3)                      */
#define IPP_LN2   ( 0.69314718055994530942 )  /* ln(2)                        */
#define IPP_LN3   ( 1.09861228866810969139 )  /* ln(3)                        */
#define IPP_E     ( 2.71828182845904523536 )  /* e                            */
#define IPP_RE    ( 0.36787944117144232159 )  /* 1/e                          */
#define IPP_EPS23 ( 1.19209289e-07f )
#define IPP_EPS52 ( 2.2204460492503131e-016 )

#define IPP_MAX_8U     ( 0xFF )
#define IPP_MAX_16U    ( 0xFFFF )
#define IPP_MAX_32U    ( 0xFFFFFFFF )
#define IPP_MIN_8U     ( 0 )
#define IPP_MIN_16U    ( 0 )
#define IPP_MIN_32U    ( 0 )
#define IPP_MIN_8S     (-128 )
#define IPP_MAX_8S     ( 127 )
#define IPP_MIN_16S    (-32768 )
#define IPP_MAX_16S    ( 32767 )
#define IPP_MIN_32S    (-2147483647 - 1 )
#define IPP_MAX_32S    ( 2147483647 )
#define IPP_MIN_64U    ( 0 )

#if defined( _WIN32 ) || defined ( _WIN64 )
  #define IPP_MAX_64S  ( 9223372036854775807i64 )
  #define IPP_MIN_64S  (-9223372036854775807i64 - 1 )
  #define IPP_MAX_64U  ( 0xffffffffffffffffL ) /* 18446744073709551615 */
#else
  #define IPP_MAX_64S  ( 9223372036854775807LL )
  #define IPP_MIN_64S  (-9223372036854775807LL - 1 )
  #define IPP_MAX_64U  ( 0xffffffffffffffffLL ) /* 18446744073709551615 */
#endif

#define IPP_MINABS_32F ( 1.175494351e-38f )
#define IPP_MAXABS_32F ( 3.402823466e+38f )
#define IPP_EPS_32F    ( 1.192092890e-07f )
#define IPP_MINABS_64F ( 2.2250738585072014e-308 )
#define IPP_MAXABS_64F ( 1.7976931348623158e+308 )
#define IPP_EPS_64F    ( 2.2204460492503131e-016 )

#define IPP_DEG_TO_RAD( deg ) ( (deg)/180.0 * IPP_PI )
#define IPP_COUNT_OF( obj )  (sizeof(obj)/sizeof(obj[0]))

#define IPP_MAX( a, b ) ( ((a) > (b)) ? (a) : (b) )
#define IPP_MIN( a, b ) ( ((a) < (b)) ? (a) : (b) )

#define IPP_ABS( a ) ( ((a) < 0) ? (-(a)) : (a) )

#define IPP_TEMPORAL_COPY      0x0
#define IPP_NONTEMPORAL_STORE  0x01
#define IPP_NONTEMPORAL_LOAD   0x02

#if !defined( _OWN_BLDPCS )
typedef enum {
 /* Enumeration:               Processor:                                                         */
    ippCpuUnknown  = 0x00,
    ippCpuPP       = 0x01, /* Intel(R) Pentium(R) processor                                       */
    ippCpuPMX      = 0x02, /* Intel(R) Pentium(R) processor with MMX(TM) technology                        */
    ippCpuPPR      = 0x03, /* Intel(R) Pentium(R) Pro processor                                            */
    ippCpuPII      = 0x04, /* Intel(R) Pentium(R) II processor                                             */
    ippCpuPIII     = 0x05, /* Intel(R) Pentium(R) III processor and Pentium(R) III Xeon(R) processor       */
    ippCpuP4       = 0x06, /* Intel(R) Pentium(R) 4 processor and Intel(R) Xeon(R) processor               */
    ippCpuP4HT     = 0x07, /* Intel(R) Pentium(R) 4 Processor with HT Technology                           */
    ippCpuP4HT2    = 0x08, /* Intel(R) Pentium(R) 4 processor with Intel(R) Streaming SIMD Extensions 3    */
    ippCpuCentrino = 0x09, /* Intel(R) Centrino(R) mobile technology                              */
    ippCpuCoreSolo = 0x0a, /* Intel(R) Core(TM) Solo processor                                    */
    ippCpuCoreDuo  = 0x0b, /* Intel(R) Core(TM) Duo processor                                     */
    ippCpuITP      = 0x10, /* Intel(R) Itanium(R) processor                                       */
    ippCpuITP2     = 0x11, /* Intel(R) Itanium(R) 2 processor                                     */
    ippCpuEM64T    = 0x20, /* Intel(R) 64 Instruction Set Architecture (ISA)                      */
    ippCpuC2D      = 0x21, /* Intel(R) Core(TM) 2 Duo processor                                   */
    ippCpuC2Q      = 0x22, /* Intel(R) Core(TM) 2 Quad processor                                  */
    ippCpuPenryn   = 0x23, /* Intel(R) Core(TM) 2 processor with Intel(R) SSE4.1                  */
    ippCpuBonnell  = 0x24, /* Intel(R) Atom(TM) processor                                         */
    ippCpuNehalem  = 0x25, /* Intel(R) Core(TM) i7 processor                                      */
    ippCpuNext     = 0x26,
    ippCpuSSE      = 0x40, /* Processor supports Intel(R) Streaming SIMD Extensions (Intel(R) SSE) instruction set      */
    ippCpuSSE2     = 0x41, /* Processor supports Intel(R) Streaming SIMD Extensions 2 (Intel(R) SSE2) instruction set    */
    ippCpuSSE3     = 0x42, /* Processor supports Intel(R) Streaming SIMD Extensions 3 (Intel(R) SSE3) instruction set    */
    ippCpuSSSE3    = 0x43, /* Processor supports Intel(R) Supplemental Streaming SIMD Extension 3 (Intel(R) SSSE3) instruction set */
    ippCpuSSE41    = 0x44, /* Processor supports Intel(R) Streaming SIMD Extensions 4.1 (Intel(R) SSE4.1) instruction set  */
    ippCpuSSE42    = 0x45, /* Processor supports Intel(R) Streaming SIMD Extensions 4.2 (Intel(R) SSE4.2) instruction set  */
    ippCpuAVX      = 0x46, /* Processor supports Intel(R) Advanced Vector Extensions (Intel(R) AVX) instruction set     */
    ippCpuAES      = 0x47, /* Processor supports Intel(R) AES New Instructions                           */
    ippCpuSHA      = 0x48, /* Processor supports Intel(R) SHA New Instructions                           */
    ippCpuF16RND   = 0x49, /* Processor supports RDRRAND & Float16 instructions                   */
    ippCpuAVX2     = 0x4a, /* Processor supports Intel(R) Advanced Vector Extensions 2 (Intel(R) AVX2) instruction set   */
    ippCpuADCOX    = 0x4b, /* Processor supports ADCX and ADOX instructions                       */
    ippCpuX8664    = 0x60  /* Processor supports 64 bit extension                                 */
} IppCpuType;

#define   ippCPUID_MMX        0x00000001   /* Intel Architecture MMX technology supported  */
#define   ippCPUID_SSE        0x00000002   /* Streaming SIMD Extensions                    */
#define   ippCPUID_SSE2       0x00000004   /* Streaming SIMD Extensions 2                  */
#define   ippCPUID_SSE3       0x00000008   /* Streaming SIMD Extensions 3                  */
#define   ippCPUID_SSSE3      0x00000010   /* Supplemental Streaming SIMD Extensions 3     */
#define   ippCPUID_MOVBE      0x00000020   /* The processor supports MOVBE instruction     */
#define   ippCPUID_SSE41      0x00000040   /* Streaming SIMD Extensions 4.1                */
#define   ippCPUID_SSE42      0x00000080   /* Streaming SIMD Extensions 4.2                */
#define   ippCPUID_AVX        0x00000100   /* Advanced Vector Extensions instruction set   */
#define   ippAVX_ENABLEDBYOS  0x00000200   /* The operating system supports AVX            */
#define   ippCPUID_AES        0x00000400   /* AES instruction                              */
#define   ippCPUID_CLMUL      0x00000800   /* PCLMULQDQ instruction                        */
#define   ippCPUID_ABR        0x00001000   /* Reserved                                     */
#define   ippCPUID_RDRAND     0x00002000   /* Read Random Number instructions              */
#define   ippCPUID_F16C       0x00004000   /* Float16 instructions                         */
#define   ippCPUID_AVX2       0x00008000   /* Advanced Vector Extensions 2 instruction set */
#define   ippCPUID_ADCOX      0x00010000   /* ADCX and ADOX instructions                   */
#define   ippCPUID_RDSEED     0x00020000   /* The RDSEED instruction                       */
#define   ippCPUID_PREFETCHW  0x00040000   /* The PREFETCHW instruction                    */
#define   ippCPUID_SHA        0x00080000   /* Intel (R) SHA Extensions                     */
#define   ippCPUID_KNC        0x80000000   /* Intel(R) Xeon Phi(TM) Coprocessor            */

#define   ippCPUID_GETINFO_A  0x616f666e69746567

typedef struct {
    int    major;                     /* e.g. 1                               */
    int    minor;                     /* e.g. 2                               */
    int    majorBuild;                /* e.g. 3                               */
    int    build;                     /* e.g. 10, always >= majorBuild        */
    char  targetCpu[4];               /* corresponding to Intel(R) processor  */
    const char* Name;                 /* e.g. "ippsw7"                        */
    const char* Version;              /* e.g. "v1.2 Beta"                     */
    const char* BuildDate;            /* e.g. "Jul 20 99"                     */
} IppLibraryVersion;


typedef unsigned char   Ipp8u;
typedef unsigned short  Ipp16u;
typedef unsigned int    Ipp32u;

typedef signed char    Ipp8s;
typedef signed short   Ipp16s;
typedef signed int     Ipp32s;
typedef float   Ipp32f;
typedef __INT64 Ipp64s;
typedef __UINT64 Ipp64u;
typedef double  Ipp64f;

typedef struct {
    Ipp8s  re;
    Ipp8s  im;
} Ipp8sc;

typedef struct {
    Ipp16s  re;
    Ipp16s  im;
} Ipp16sc;

typedef struct {
    Ipp16u  re;
    Ipp16u  im;
} Ipp16uc;

typedef struct {
    Ipp32s  re;
    Ipp32s  im;
} Ipp32sc;

typedef struct {
    Ipp32f  re;
    Ipp32f  im;
} Ipp32fc;

typedef struct {
    Ipp64s  re;
    Ipp64s  im;
} Ipp64sc;

typedef struct {
    Ipp64f  re;
    Ipp64f  im;
} Ipp64fc;

typedef int IppEnum;

typedef enum {
    ippRndZero,
    ippRndNear,
    ippRndFinancial
} IppRoundMode;


typedef enum {
    ippAlgHintNone,
    ippAlgHintFast,
    ippAlgHintAccurate
} IppHintAlgorithm;

typedef enum {
    ippCmpLess,
    ippCmpLessEq,
    ippCmpEq,
    ippCmpGreaterEq,
    ippCmpGreater
} IppCmpOp;

typedef enum {
    ippAlgAuto    = 0x00000000,
    ippAlgDirect  = 0x00000001,
    ippAlgFFT     = 0x00000002,
    ippAlgMask    = 0x000000FF
} IppAlgType;

typedef enum {
    ippsNormNone  = 0x00000000, /* default */
    ippsNormA     = 0x00000100, /* biased normalization */
    ippsNormB     = 0x00000200, /* unbiased normalization */
    ippsNormMask  = 0x0000FF00
} IppsNormOp;

typedef enum {
    ippiNormNone        = 0x00000000, /* default */
    ippiNorm            = 0x00000100, /* normalized form */
    ippiNormCoefficient = 0x00000200, /* correlation coefficient in the range [-1.0 ... 1.0] */
    ippiNormMask        = 0x0000FF00
} IppiNormOp;

typedef enum {
    ippNormInf  =   0x00000001,
    ippNormL1   =   0x00000002,
    ippNormL2   =   0x00000004
} IppNormType;


typedef enum {
   ippiROIFull   = 0x00000000,
   ippiROIValid  = 0x00010000,
   ippiROISame   = 0x00020000,
   ippiROIMask   = 0x00FF0000
} IppiROIShape;

enum {
    IPP_FFT_DIV_FWD_BY_N = 1,
    IPP_FFT_DIV_INV_BY_N = 2,
    IPP_FFT_DIV_BY_SQRTN = 4,
    IPP_FFT_NODIV_BY_ANY = 8
};

enum {
    IPP_DIV_FWD_BY_N = 1,
    IPP_DIV_INV_BY_N = 2,
    IPP_DIV_BY_SQRTN = 4,
    IPP_NODIV_BY_ANY = 8
};

typedef enum {
   ippUndef = -1,
   ipp1u    =  0,
   ipp8u    =  1,
   ipp8uc   =  2,
   ipp8s    =  3,
   ipp8sc   =  4,
   ipp16u   =  5,
   ipp16uc  =  6,
   ipp16s   =  7,
   ipp16sc  =  8,
   ipp32u   =  9,
   ipp32uc  = 10,
   ipp32s   = 11,
   ipp32sc  = 12,
   ipp32f   = 13,
   ipp32fc  = 14,
   ipp64u   = 15,
   ipp64uc  = 16,
   ipp64s   = 17,
   ipp64sc  = 18,
   ipp64f   = 19,
   ipp64fc  = 20
} IppDataType;

typedef enum {
   ippC0    =  0,
   ippC1    =  1,
   ippC2    =  2,
   ippC3    =  3,
   ippC4    =  4,
   ippP2    =  5,
   ippP3    =  6,
   ippP4    =  7,
   ippAC1   =  8,
   ippAC4   =  9,
   ippA0C4  = 10,
   ippAP4   = 11
} IppChannels;

typedef enum _IppiBorderType {
    ippBorderConst     =  0,
    ippBorderRepl      =  1,
    ippBorderWrap      =  2,
    ippBorderMirror    =  3, /* left border: 012... -> 21012... */
    ippBorderMirrorR   =  4, /* left border: 012... -> 210012... */
    ippBorderInMem     =  6,
    ippBorderTransp    =  7,
    ippBorderInMemTop     =  0x0010,
    ippBorderInMemBottom  =  0x0020,
    ippBorderInMemLeft    =  0x0040,
    ippBorderInMemRight   =  0x0080
} IppiBorderType;

typedef enum {
    ippAxsHorizontal,
    ippAxsVertical,
    ippAxsBoth,
    ippAxs45,
    ippAxs135
} IppiAxis;

typedef struct {
    int x;
    int y;
    int width;
    int height;
} IppiRect;

typedef struct {
    int x;
    int y;
} IppiPoint;

typedef struct {
    int width;
    int height;
} IppiSize;

typedef struct {
    Ipp32f x;
    Ipp32f y;
} IppiPoint_32f;

typedef struct {
    Ipp32f rho;
    Ipp32f theta;
} IppPointPolar;

#define DECLARE_IPPCONTEXT(IppCtxName) struct __##IppCtxName##__; typedef struct __##IppCtxName##__ IppCtxName

struct VLCDecodeSpec_32s;
typedef struct VLCDecodeSpec_32s IppsVLCDecodeSpec_32s;
struct VLCEncodeSpec_32s;
typedef struct VLCEncodeSpec_32s IppsVLCEncodeSpec_32s;
struct VLCDecodeUTupleSpec_32s;
typedef struct VLCDecodeUTupleSpec_32s IppsVLCDecodeUTupleSpec_32s;

enum {
     IPP_UPPER        = 1,
     IPP_LEFT         = 2,
     IPP_CENTER       = 4,
     IPP_RIGHT        = 8,
     IPP_LOWER        = 16,
     IPP_UPPER_LEFT   = 32,
     IPP_UPPER_RIGHT  = 64,
     IPP_LOWER_LEFT   = 128,
     IPP_LOWER_RIGHT  = 256
};

typedef enum  _IppiMaskSize {
    ippMskSize1x3 = 13,
    ippMskSize1x5 = 15,
    ippMskSize3x1 = 31,
    ippMskSize3x3 = 33,
    ippMskSize5x1 = 51,
    ippMskSize5x5 = 55
} IppiMaskSize;

enum {
    IPPI_INTER_NN     = 1,
    IPPI_INTER_LINEAR = 2,
    IPPI_INTER_CUBIC  = 4,
    IPPI_INTER_CUBIC2P_BSPLINE,     /* two-parameter cubic filter (B=1, C=0) */
    IPPI_INTER_CUBIC2P_CATMULLROM,  /* two-parameter cubic filter (B=0, C=1/2) */
    IPPI_INTER_CUBIC2P_B05C03,      /* two-parameter cubic filter (B=1/2, C=3/10) */
    IPPI_INTER_SUPER  = 8,
    IPPI_INTER_LANCZOS = 16,
    IPPI_ANTIALIASING  = (1 << 29),
    IPPI_SUBPIXEL_EDGE = (1 << 30),
    IPPI_SMOOTH_EDGE   = (1 << 31)
};

typedef enum {
    ippPolyphase_1_2,
    ippPolyphase_3_5,
    ippPolyphase_2_3,
    ippPolyphase_7_10,
    ippPolyphase_3_4
} IppiFraction;

typedef enum _IppiDifferentialKernel
{
    ippFilterSobelVert,
    ippFilterSobelHoriz,
    ippFilterSobel,
    ippFilterScharrVert,
    ippFilterScharrHoriz,
    ippFilterScharr,
    ippFilterCentralDiffVert,
    ippFilterCentralDiffHoriz,
    ippFilterCentralDiff,
}IppiDifferentialKernel;

enum {
    IPP_FASTN_ORIENTATION = 0x0001,
    IPP_FASTN_NMS         = 0x0002,
    IPP_FASTN_CIRCLE      = 0X0004,
    IPP_FASTN_SCORE_MODE0 = 0X0020
};

typedef enum { ippFalse = 0, ippTrue = 1 } IppBool;

typedef enum {ippWinBartlett,ippWinBlackman,ippWinHamming,ippWinHann,ippWinRect} IppWinType;

typedef enum { ippButterworth, ippChebyshev1 } IppsIIRFilterType;

typedef enum  { ippZCR=0,   ippZCXor,   ippZCC } IppsZCType;

typedef struct {
    int width;
    int height;
    int depth;
} IpprVolume;

typedef struct {
    int x;
    int y;
    int z;
    int width;
    int height;
    int depth;
} IpprCuboid;

typedef struct {
    int x;
    int y;
    int z;
} IpprPoint;


/* /////////////////////////////////////////////////////////////////////////////
//        The following enumerator defines a status of IPP operations
//                     negative value means error
*/
typedef enum {
    /* errors */
    ippStsNotSupportedModeErr    = -9999,/* The requested mode is currently not supported.  */
    ippStsCpuNotSupportedErr     = -9998,/* The target CPU is not supported. */
    ippStsInplaceModeNotSupportedErr = -9997,/* The inplace operation is currently not supported. */

    ippStsWarpDirectionErr       = -231, /* The warp transform direction is illegal */

    ippStsFilterTypeErr          = -230, /* The filter type is incorrect or not supported */

    ippStsNormErr                = -229, /* The norm is incorrect or not supported */

    ippStsAlgTypeErr             = -228, /* Algorithm type is not supported.        */
    ippStsMisalignedOffsetErr    = -227, /* The offset is not aligned with an element. */

    ippStsQuadraticNonResidueErr = -226, /* SQRT operation on quadratic non-residue value. */

    ippStsBorderErr              = -225, /* Illegal value for border type.*/

    ippStsDitherTypeErr          = -224, /* Dithering type is not supported. */
    ippStsH264BufferFullErr      = -223, /* Buffer for the output bitstream is full. */
    ippStsWrongAffinitySettingErr= -222, /* An affinity setting does not correspond to the affinity setting that was set by f.ippSetAffinity(). */
    ippStsLoadDynErr             = -221, /* Error when loading the dynamic library. */

    ippStsPointAtInfinity        = -220, /* Point at infinity is detected.  */

    ippStsI18nUnsupportedErr     = -219, /* Internationalization (i18n) is not supported.                                                          */
    ippStsI18nMsgCatalogOpenErr  = -218, /* Message catalog cannot be opened, for more info use errno on Linux* OS and GetLastError on Windows* OS.*/
    ippStsI18nMsgCatalogCloseErr = -217, /* Message catalog cannot be closed, for more info use errno on Linux* OS and GetLastError on Windows* OS.*/

    ippStsUnknownStatusCodeErr   = -216, /* Unknown status code. */

    ippStsOFBSizeErr             = -215, /* Incorrect value for crypto OFB block size. */
    ippStsLzoBrokenStreamErr     = -214, /* LZO safe decompression function cannot decode LZO stream. */

    ippStsRoundModeNotSupportedErr  = -213, /* Rounding mode is not supported. */
    ippStsDecimateFractionErr    = -212, /* Fraction in Decimate is not supported. */
    ippStsWeightErr              = -211, /* Incorrect value for weight. */

    ippStsQualityIndexErr        = -210, /* Cannot calculate the quality index for an image filled with a constant. */
    ippStsIIRPassbandRippleErr   = -209, /* Ripple in passband for Chebyshev1 design is less than zero, equal to zero, or greater than 29. */
    ippStsFilterFrequencyErr     = -208, /* Cutoff frequency of filter is less than zero, equal to zero, or greater than 0.5. */
    ippStsFIRGenOrderErr         = -207, /* Order of the FIR filter for design is less than 1.                    */
    ippStsIIRGenOrderErr         = -206, /* Order of the IIR filter for design is less than 1, or greater than 12. */

    ippStsConvergeErr            = -205, /* The algorithm does not converge. */
    ippStsSizeMatchMatrixErr     = -204, /* The sizes of the source matrices are unsuitable. */
    ippStsCountMatrixErr         = -203, /* Count value is less than, or equal to zero. */
    ippStsRoiShiftMatrixErr      = -202, /* RoiShift value is negative or not divisible by the size of the data type. */

    ippStsResizeNoOperationErr   = -201, /* One of the output image dimensions is less than 1 pixel. */
    ippStsSrcDataErr             = -200, /* The source buffer contains unsupported data. */
    ippStsMaxLenHuffCodeErr      = -199, /* Huff: Max length of Huffman code is more than the expected one. */
    ippStsCodeLenTableErr        = -198, /* Huff: Invalid codeLenTable. */
    ippStsFreqTableErr           = -197, /* Huff: Invalid freqTable. */

    ippStsIncompleteContextErr   = -196, /* Crypto: set up of context is not complete. */

    ippStsSingularErr            = -195, /* Matrix is singular. */
    ippStsSparseErr              = -194, /* Positions of taps are not in ascending order, or are negative, or repetitive. */
    ippStsBitOffsetErr           = -193, /* Incorrect bit offset value. */
    ippStsQPErr                  = -192, /* Incorrect quantization parameter value. */
    ippStsVLCErr                 = -191, /* Illegal VLC or FLC is detected during stream decoding. */
    ippStsRegExpOptionsErr       = -190, /* RegExp: Options for the pattern are incorrect. */
    ippStsRegExpErr              = -189, /* RegExp: The structure pRegExpState contains incorrect data. */
    ippStsRegExpMatchLimitErr    = -188, /* RegExp: The match limit is exhausted. */
    ippStsRegExpQuantifierErr    = -187, /* RegExp: Incorrect quantifier. */
    ippStsRegExpGroupingErr      = -186, /* RegExp: Incorrect grouping. */
    ippStsRegExpBackRefErr       = -185, /* RegExp: Incorrect back reference. */
    ippStsRegExpChClassErr       = -184, /* RegExp: Incorrect character class. */
    ippStsRegExpMetaChErr        = -183, /* RegExp: Incorrect metacharacter. */
    ippStsStrideMatrixErr        = -182,  /* Stride value is not positive or not divisible by the size of the data type. */
    ippStsCTRSizeErr             = -181,  /* Incorrect value for crypto CTR block size. */
    ippStsJPEG2KCodeBlockIsNotAttached =-180, /* Codeblock parameters are not attached to the state structure. */
    ippStsNotPosDefErr           = -179,      /* Matrix is not positive definite. */

    ippStsEphemeralKeyErr        = -178, /* ECC: Invalid ephemeral key.   */
    ippStsMessageErr             = -177, /* ECC: Invalid message digest.  */
    ippStsShareKeyErr            = -176, /* ECC: Invalid share key.   */
    ippStsIvalidPublicKey        = -175, /* ECC: Invalid public key.  */
    ippStsIvalidPrivateKey       = -174, /* ECC: Invalid private key. */
    ippStsOutOfECErr             = -173, /* ECC: Point out of EC.     */
    ippStsECCInvalidFlagErr      = -172, /* ECC: Invalid Flag.        */

    ippStsMP3FrameHeaderErr      = -171,  /* Error in fields of the IppMP3FrameHeader structure. */
    ippStsMP3SideInfoErr         = -170,  /* Error in fields of the IppMP3SideInfo structure. */

    ippStsBlockStepErr           = -169,  /* Step for Block is less than 8. */
    ippStsMBStepErr              = -168,  /* Step for MB is less than 16. */

    ippStsAacPrgNumErr           = -167,  /* AAC: Invalid number of elements for one program.   */
    ippStsAacSectCbErr           = -166,  /* AAC: Invalid section codebook.                     */
    ippStsAacSfValErr            = -164,  /* AAC: Invalid scalefactor value.                    */
    ippStsAacCoefValErr          = -163,  /* AAC: Invalid quantized coefficient value.          */
    ippStsAacMaxSfbErr           = -162,  /* AAC: Invalid coefficient index.  */
    ippStsAacPredSfbErr          = -161,  /* AAC: Invalid predicted coefficient index.  */
    ippStsAacPlsDataErr          = -160,  /* AAC: Invalid pulse data attributes.  */
    ippStsAacGainCtrErr          = -159,  /* AAC: Gain control is not supported.  */
    ippStsAacSectErr             = -158,  /* AAC: Invalid number of sections.  */
    ippStsAacTnsNumFiltErr       = -157,  /* AAC: Invalid number of TNS filters.  */
    ippStsAacTnsLenErr           = -156,  /* AAC: Invalid length of TNS region.  */
    ippStsAacTnsOrderErr         = -155,  /* AAC: Invalid order of TNS filter.  */
    ippStsAacTnsCoefResErr       = -154,  /* AAC: Invalid bit-resolution for TNS filter coefficients.  */
    ippStsAacTnsCoefErr          = -153,  /* AAC: Invalid coefficients of TNS filter. */
    ippStsAacTnsDirectErr        = -152,  /* AAC: Invalid direction TNS filter.  */
    ippStsAacTnsProfileErr       = -151,  /* AAC: Invalid TNS profile.  */
    ippStsAacErr                 = -150,  /* AAC: Internal error.  */
    ippStsAacBitOffsetErr        = -149,  /* AAC: Invalid current bit offset in bitstream.  */
    ippStsAacAdtsSyncWordErr     = -148,  /* AAC: Invalid ADTS syncword.  */
    ippStsAacSmplRateIdxErr      = -147,  /* AAC: Invalid sample rate index.  */
    ippStsAacWinLenErr           = -146,  /* AAC: Invalid window length (not short or long).  */
    ippStsAacWinGrpErr           = -145,  /* AAC: Invalid number of groups for current window length.  */
    ippStsAacWinSeqErr           = -144,  /* AAC: Invalid window sequence range.  */
    ippStsAacComWinErr           = -143,  /* AAC: Invalid common window flag.  */
    ippStsAacStereoMaskErr       = -142,  /* AAC: Invalid stereo mask.  */
    ippStsAacChanErr             = -141,  /* AAC: Invalid channel number.  */
    ippStsAacMonoStereoErr       = -140,  /* AAC: Invalid mono-stereo flag.  */
    ippStsAacStereoLayerErr      = -139,  /* AAC: Invalid this Stereo Layer flag.  */
    ippStsAacMonoLayerErr        = -138,  /* AAC: Invalid this Mono Layer flag.  */
    ippStsAacScalableErr         = -137,  /* AAC: Invalid scalable object flag.  */
    ippStsAacObjTypeErr          = -136,  /* AAC: Invalid audio object type.  */
    ippStsAacWinShapeErr         = -135,  /* AAC: Invalid window shape.  */
    ippStsAacPcmModeErr          = -134,  /* AAC: Invalid PCM output interleaving indicator.  */
    ippStsVLCUsrTblHeaderErr          = -133,  /* VLC: Invalid header inside table. */
    ippStsVLCUsrTblUnsupportedFmtErr  = -132,  /* VLC: Table format is not supported.  */
    ippStsVLCUsrTblEscAlgTypeErr      = -131,  /* VLC: Ecs-algorithm is not supported. */
    ippStsVLCUsrTblEscCodeLengthErr   = -130,  /* VLC: Esc-code length inside table header is incorrect. */
    ippStsVLCUsrTblCodeLengthErr      = -129,  /* VLC: Code length inside table is incorrect.  */
    ippStsVLCInternalTblErr           = -128,  /* VLC: Invalid internal table. */
    ippStsVLCInputDataErr             = -127,  /* VLC: Invalid input data. */
    ippStsVLCAACEscCodeLengthErr      = -126,  /* VLC: Invalid AAC-Esc code length. */
    ippStsNoiseRangeErr         = -125,  /* Noise value for Wiener Filter is out of range. */
    ippStsUnderRunErr           = -124,  /* Error in data under run. */
    ippStsPaddingErr            = -123,  /* Detected padding error indicates the possible data corruption. */
    ippStsCFBSizeErr            = -122,  /* Incorrect value for crypto CFB block size. */
    ippStsPaddingSchemeErr      = -121,  /* Invalid padding scheme.  */
    ippStsInvalidCryptoKeyErr   = -120,  /* A compromised key causes suspansion of the requested cryptographic operation.  */
    ippStsLengthErr             = -119,  /* Incorrect value for string length. */
    ippStsBadModulusErr         = -118,  /* Bad modulus caused a failure in module inversion. */
    ippStsLPCCalcErr            = -117,  /* Cannot evaluate linear prediction. */
    ippStsRCCalcErr             = -116,  /* Cannot compute reflection coefficients. */
    ippStsIncorrectLSPErr       = -115,  /* Incorrect values for Linear Spectral Pair. */
    ippStsNoRootFoundErr        = -114,  /* No roots are found for equation. */
    ippStsJPEG2KBadPassNumber   = -113,  /* Pass number exceeds allowed boundaries [0,nOfPasses-1]. */
    ippStsJPEG2KDamagedCodeBlock= -112,  /* Codeblock for decoding contains damaged data. */
    ippStsH263CBPYCodeErr       = -111,  /* Illegal Huffman code is detected through CBPY stream processing. */
    ippStsH263MCBPCInterCodeErr = -110,  /* Illegal Huffman code is detected through MCBPC Inter stream processing. */
    ippStsH263MCBPCIntraCodeErr = -109,  /* Illegal Huffman code is detected through MCBPC Intra stream processing. */
    ippStsNotEvenStepErr        = -108,  /* Step value is not pixel multiple. */
    ippStsHistoNofLevelsErr     = -107,  /* Number of levels for histogram is less than 2. */
    ippStsLUTNofLevelsErr       = -106,  /* Number of levels for LUT is less than 2. */
    ippStsMP4BitOffsetErr       = -105,  /* Incorrect bit offset value. */
    ippStsMP4QPErr              = -104,  /* Incorrect quantization parameter. */
    ippStsMP4BlockIdxErr        = -103,  /* Incorrect block index. */
    ippStsMP4BlockTypeErr       = -102,  /* Incorrect block type. */
    ippStsMP4MVCodeErr          = -101,  /* Illegal Huffman code is detected during MV stream processing. */
    ippStsMP4VLCCodeErr         = -100,  /* Illegal Huffman code is detected during VLC stream processing. */
    ippStsMP4DCCodeErr          = -99,   /* Illegal code is detected during DC stream processing. */
    ippStsMP4FcodeErr           = -98,   /* Incorrect fcode value. */
    ippStsMP4AlignErr           = -97,   /* Incorrect buffer alignment .           */
    ippStsMP4TempDiffErr        = -96,   /* Incorrect temporal difference.         */
    ippStsMP4BlockSizeErr       = -95,   /* Incorrect size of a block or macroblock. */
    ippStsMP4ZeroBABErr         = -94,   /* All BAB values are equal to zero.             */
    ippStsMP4PredDirErr         = -93,   /* Incorrect prediction direction.        */
    ippStsMP4BitsPerPixelErr    = -92,   /* Incorrect number of bits per pixel.    */
    ippStsMP4VideoCompModeErr   = -91,   /* Incorrect video component mode.       */
    ippStsMP4LinearModeErr      = -90,   /* Incorrect DC linear mode. */
    ippStsH263PredModeErr       = -83,   /* Incorrect Prediction Mode value.                                       */
    ippStsH263BlockStepErr      = -82,   /* The step value is less than 8.                                         */
    ippStsH263MBStepErr         = -81,   /* The step value is less than 16.                                        */
    ippStsH263FrameWidthErr     = -80,   /* The frame width is less than 8.                                        */
    ippStsH263FrameHeightErr    = -79,   /* The frame height is less than, or equal to zero.                        */
    ippStsH263ExpandPelsErr     = -78,   /* Expand pixels number is less than 8.                               */
    ippStsH263PlaneStepErr      = -77,   /* Step value is less than the plane width.                           */
    ippStsH263QuantErr          = -76,   /* Quantizer value is less than, or equal to zero, or greater than 31. */
    ippStsH263MVCodeErr         = -75,   /* Illegal Huffman code is detected during MV stream processing.                  */
    ippStsH263VLCCodeErr        = -74,   /* Illegal Huffman code is detected during VLC stream processing.                 */
    ippStsH263DCCodeErr         = -73,   /* Illegal code is detected during DC stream processing.                          */
    ippStsH263ZigzagLenErr      = -72,   /* Zigzag compact length is more than 64.                             */
    ippStsFBankFreqErr          = -71,   /* Incorrect value for the filter bank frequency parameter. */
    ippStsFBankFlagErr          = -70,   /* Incorrect value for the filter bank parameter.           */
    ippStsFBankErr              = -69,   /* Filter bank is not correctly initialized.              */
    ippStsNegOccErr             = -67,   /* Occupation count is negative.                     */
    ippStsCdbkFlagErr           = -66,   /* Incorrect value for the codebook flag parameter. */
    ippStsSVDCnvgErr            = -65,   /* SVD algorithm does not converge.               */
    ippStsJPEGHuffTableErr      = -64,   /* JPEG Huffman table is destroyed.        */
    ippStsJPEGDCTRangeErr       = -63,   /* JPEG DCT coefficient is out of range. */
    ippStsJPEGOutOfBufErr       = -62,   /* Attempt to access out of the buffer limits.   */
    ippStsDrawTextErr           = -61,   /* System error in the draw text operation. */
    ippStsChannelOrderErr       = -60,   /* Incorrect order of the destination channels. */
    ippStsZeroMaskValuesErr     = -59,   /* All values of the mask are equal to zero. */
    ippStsQuadErr               = -58,   /* The quadrangle is nonconvex or degenerates into triangle, line, or point */
    ippStsRectErr               = -57,   /* Size of the rectangle region is less than, or equal to 1. */
    ippStsCoeffErr              = -56,   /* Incorrect values for transformation coefficients.   */
    ippStsNoiseValErr           = -55,   /* Incorrect value for noise amplitude for dithering.             */
    ippStsDitherLevelsErr       = -54,   /* Number of dithering levels is out of range.             */
    ippStsNumChannelsErr        = -53,   /* Number of channels is incorrect, or not supported.                  */
    ippStsCOIErr                = -52,   /* COI is out of range. */
    ippStsDivisorErr            = -51,   /* Divisor is equal to zero, function is aborted. */
    ippStsAlphaTypeErr          = -50,   /* Illegal type of image compositing operation.                           */
    ippStsGammaRangeErr         = -49,   /* Gamma range bounds is less than, or equal to zero.                      */
    ippStsGrayCoefSumErr        = -48,   /* Sum of the conversion coefficients must be less than, or equal to 1.    */
    ippStsChannelErr            = -47,   /* Illegal channel number.                                                */
    ippStsToneMagnErr           = -46,   /* Tone magnitude is less than, or equal to zero.                          */
    ippStsToneFreqErr           = -45,   /* Tone frequency is negative, or greater than, or equal to 0.5.           */
    ippStsTonePhaseErr          = -44,   /* Tone phase is negative, or greater than, or equal to 2*PI.              */
    ippStsTrnglMagnErr          = -43,   /* Triangle magnitude is less than, or equal to zero.                      */
    ippStsTrnglFreqErr          = -42,   /* Triangle frequency is negative, or greater than, or equal to 0.5.       */
    ippStsTrnglPhaseErr         = -41,   /* Triangle phase is negative, or greater than, or equal to 2*PI.          */
    ippStsTrnglAsymErr          = -40,   /* Triangle asymmetry is less than -PI, or greater than, or equal to PI.   */
    ippStsHugeWinErr            = -39,   /* Kaiser window is too big.                                             */
    ippStsJaehneErr             = -38,   /* Magnitude value is negative.                                           */
    ippStsStrideErr             = -37,   /* Stride value is less than the length of the row. */
    ippStsEpsValErr             = -36,   /* Negative epsilon value.             */
    ippStsWtOffsetErr           = -35,   /* Invalid offset value for wavelet filter.                                       */
    ippStsAnchorErr             = -34,   /* Anchor point is outside the mask.                                             */
    ippStsMaskSizeErr           = -33,   /* Invalid mask size.                                                           */
    ippStsShiftErr              = -32,   /* Shift value is less than zero.                                                */
    ippStsSampleFactorErr       = -31,   /* Sampling factor is less than, or equal to zero.                                */
    ippStsSamplePhaseErr        = -30,   /* Phase value is out of range: 0 <= phase < factor.                             */
    ippStsFIRMRFactorErr        = -29,   /* MR FIR sampling factor is less than, or equal to zero.                         */
    ippStsFIRMRPhaseErr         = -28,   /* MR FIR sampling phase is negative, or greater than, or equal to the sampling factor. */
    ippStsRelFreqErr            = -27,   /* Relative frequency value is out of range.                                     */
    ippStsFIRLenErr             = -26,   /* Length of a FIR filter is less than, or equal to zero.                         */
    ippStsIIROrderErr           = -25,   /* Order of an IIR filter is not valid. */
    ippStsDlyLineIndexErr       = -24,   /* Invalid value for the delay line sample index. */
    ippStsResizeFactorErr       = -23,   /* Resize factor(s) is less than, or equal to zero. */
    ippStsInterpolationErr      = -22,   /* Invalid interpolation mode. */
    ippStsMirrorFlipErr         = -21,   /* Invalid flip mode.                                         */
    ippStsMoment00ZeroErr       = -20,   /* Moment value M(0,0) is too small to continue calculations. */
    ippStsThreshNegLevelErr     = -19,   /* Negative value of the level in the threshold operation.    */
    ippStsThresholdErr          = -18,   /* Invalid threshold bounds. */
    ippStsContextMatchErr       = -17,   /* Context parameter does not match the operation. */
    ippStsFftFlagErr            = -16,   /* Invalid value for the FFT flag parameter. */
    ippStsFftOrderErr           = -15,   /* Invalid value for the FFT order parameter. */
    ippStsStepErr               = -14,   /* Step value is not valid. */
    ippStsScaleRangeErr         = -13,   /* Scale bounds are out of range. */
    ippStsDataTypeErr           = -12,   /* Data type is incorrect or not supported. */
    ippStsOutOfRangeErr         = -11,   /* Argument is out of range, or point is outside the image. */
    ippStsDivByZeroErr          = -10,   /* An attempt to divide by zero. */
    ippStsMemAllocErr           = -9,    /* Memory allocated for the operation is not enough.*/
    ippStsNullPtrErr            = -8,    /* Null pointer error. */
    ippStsRangeErr              = -7,    /* Incorrect values for bounds: the lower bound is greater than the upper bound. */
    ippStsSizeErr               = -6,    /* Incorrect value for data size. */
    ippStsBadArgErr             = -5,    /* Incorrect arg/param of the function.  */
    ippStsNoMemErr              = -4,    /* Not enough memory for the operation. */
    ippStsSAReservedErr3        = -3,    /* Unknown/unspecified error, -3. */
    ippStsErr                   = -2,    /* Unknown/unspecified error, -2. */
    ippStsSAReservedErr1        = -1,    /* Unknown/unspecified error, -1. */

     /* no errors */
    ippStsNoErr                 =   0,   /* No errors. */

     /* warnings  */
    ippStsNoOperation       =   1,       /* No operation has been executed. */
    ippStsMisalignedBuf     =   2,       /* Misaligned pointer in operation in which it must be aligned. */
    ippStsSqrtNegArg        =   3,       /* Negative value(s) for the argument in the Sqrt function. */
    ippStsInvZero           =   4,       /* INF result. Zero value was met by InvThresh with zero level. */
    ippStsEvenMedianMaskSize=   5,       /* Even size of the Median Filter mask was replaced with the odd one. */
    ippStsDivByZero         =   6,       /* Zero value(s) for the divisor in the Div function. */
    ippStsLnZeroArg         =   7,       /* Zero value(s) for the argument in the Ln function.     */
    ippStsLnNegArg          =   8,       /* Negative value(s) for the argument in the Ln function. */
    ippStsNanArg            =   9,       /* Argument value is not a number.                  */
    ippStsJPEGMarker        =   10,      /* JPEG marker in the bitstream.                 */
    ippStsResFloor          =   11,      /* All result values are floored.                        */
    ippStsOverflow          =   12,      /* Overflow in the operation.                   */
    ippStsLSFLow            =   13,      /* Quantized LP synthesis filter stability check is applied at the low boundary of [0,pi]. */
    ippStsLSFHigh           =   14,      /* Quantized LP synthesis filter stability check is applied at the high boundary of [0,pi]. */
    ippStsLSFLowAndHigh     =   15,      /* Quantized LP synthesis filter stability check is applied at both boundaries of [0,pi]. */
    ippStsZeroOcc           =   16,      /* Zero occupation count. */
    ippStsUnderflow         =   17,      /* Underflow in the operation. */
    ippStsSingularity       =   18,      /* Singularity in the operation.                                       */
    ippStsDomain            =   19,      /* Argument is out of the function domain.                                      */
    ippStsNonIntelCpu       =   20,      /* The target CPU is not Genuine Intel.                                         */
    ippStsCpuMismatch       =   21,      /* Cannot set the library for the given CPU.                                     */
    ippStsNoIppFunctionFound =  22,      /* Application does not contain Intel IPP function calls.                            */
    ippStsDllNotFoundBestUsed = 23,      /* Dispatcher cannot find the newest version of the Intel IPP dll.                  */
    ippStsNoOperationInDll  =   24,      /* The function does nothing in the dynamic version of the library.             */
    ippStsInsufficientEntropy=  25,      /* Generation of the prime/key failed due to insufficient entropy
                                            in the random seed and stimulus bit string. */
    ippStsOvermuchStrings   =   26,      /* Number of destination strings is more than expected.                         */
    ippStsOverlongString    =   27,      /* Length of one of the destination strings is more than expected.              */
    ippStsAffineQuadChanged =   28,      /* 4th vertex of destination quad is not equal to customer's one.               */
    ippStsWrongIntersectROI =   29,      /* ROI has no intersection with the source or destination ROI. No operation. */
    ippStsWrongIntersectQuad =  30,      /* Quadrangle has no intersection with the source or destination ROI. No operation. */
    ippStsSmallerCodebook   =   31,      /* Size of created codebook is less than the cdbkSize argument. */
    ippStsSrcSizeLessExpected = 32,      /* DC: Size of the source buffer is less than the expected one. */
    ippStsDstSizeLessExpected = 33,      /* DC: Size of the destination buffer is less than the expected one. */
    ippStsStreamEnd           = 34,      /* DC: The end of stream processed. */
    ippStsDoubleSize        =   35,      /* Width or height of image is odd. */
    ippStsNotSupportedCpu   =   36,      /* The CPU is not supported. */
    ippStsUnknownCacheSize  =   37,      /* The CPU is supported, but the size of the cache is unknown. */
    ippStsSymKernelExpected =   38,      /* The Kernel is not symmetric. */
    ippStsEvenMedianWeight  =   39,      /* Even weight of the Weighted Median Filter is replaced with the odd one. */
    ippStsWrongIntersectVOI =   40,      /* VOI has no intersection with the source or destination volume. No operation.                          */
    ippStsI18nMsgCatalogInvalid=41,      /* Message Catalog is invalid, English message returned.                                                 */
    ippStsI18nGetMessageFail  = 42,      /* Failed to fetch a localized message, English message returned.
                                            For more information use errno on Linux* OS and GetLastError on Windows* OS. */
    ippStsWaterfall           = 43,      /* Cannot load required library, waterfall is used. */
    ippStsPrevLibraryUsed     = 44,      /* Cannot load required library, previous dynamic library is used. */
    ippStsLLADisabled         = 45,      /* OpenMP* Low Level Affinity is disabled. */
    ippStsNoAntialiasing      = 46,      /* The mode does not support antialiasing. */
    ippStsRepetitiveSrcData   = 47,      /* DC: The source data is too repetitive. */
    ippStsSizeWrn             = 48,      /* The size does not allow to perform full operation. */
    ippStsFeatureNotSupported = 49,      /* Current CPU doesn't support at least 1 of the desired features. */
    ippStsUnknownFeature      = 50,      /* At least one of the desired features is unknown. */
    ippStsFeaturesCombination = 51       /* Wrong combination of features. */
} IppStatus;


#define ippStsOk ippStsNoErr

#endif /* _OWN_BLDPCS */


#ifdef __cplusplus
}
#endif

#endif /* __IPPDEFS_H__ */
