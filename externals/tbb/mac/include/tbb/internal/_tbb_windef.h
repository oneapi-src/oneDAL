/*
    Copyright 2005-2017 Intel Corporation.

    The source code, information and material ("Material") contained herein is owned by
    Intel Corporation or its suppliers or licensors, and title to such Material remains
    with Intel Corporation or its suppliers or licensors. The Material contains
    proprietary information of Intel or its suppliers and licensors. The Material is
    protected by worldwide copyright laws and treaty provisions. No part of the Material
    may be used, copied, reproduced, modified, published, uploaded, posted, transmitted,
    distributed or disclosed in any way without Intel's prior express written permission.
    No license under any patent, copyright or other intellectual property rights in the
    Material is granted to or conferred upon you, either expressly, by implication,
    inducement, estoppel or otherwise. Any license under such intellectual property
    rights must be express and approved by Intel in writing.

    Unless otherwise agreed by Intel in writing, you may not remove or alter this notice
    or any other notice embedded in Materials by Intel or Intel's suppliers or licensors
    in any way.
*/

#ifndef __TBB_tbb_windef_H
#error Do not #include this internal file directly; use public TBB headers instead.
#endif /* __TBB_tbb_windef_H */

// Check that the target Windows version has all API calls requried for TBB.
// Do not increase the version in condition beyond 0x0500 without prior discussion!
#if defined(_WIN32_WINNT) && _WIN32_WINNT<0x0501
#error TBB is unable to run on old Windows versions; _WIN32_WINNT must be 0x0501 or greater.
#endif

#if !defined(_MT)
#error TBB requires linkage with multithreaded C/C++ runtime library. \
       Choose multithreaded DLL runtime in project settings, or use /MD[d] compiler switch.
#endif

// Workaround for the problem with MVSC headers failing to define namespace std
namespace std {
  using ::size_t; using ::ptrdiff_t;
}

#define __TBB_STRING_AUX(x) #x
#define __TBB_STRING(x) __TBB_STRING_AUX(x)

// Default setting of TBB_USE_DEBUG
#ifdef TBB_USE_DEBUG
#    if TBB_USE_DEBUG
#        if !defined(_DEBUG)
#            pragma message(__FILE__ "(" __TBB_STRING(__LINE__) ") : Warning: Recommend using /MDd if compiling with TBB_USE_DEBUG!=0")
#        endif
#    else
#        if defined(_DEBUG)
#            pragma message(__FILE__ "(" __TBB_STRING(__LINE__) ") : Warning: Recommend using /MD if compiling with TBB_USE_DEBUG==0")
#        endif
#    endif
#endif

#if (__TBB_BUILD || __TBBMALLOC_BUILD) && !defined(__TBB_NO_IMPLICIT_LINKAGE)
#define __TBB_NO_IMPLICIT_LINKAGE 1
#endif

#if _MSC_VER
    #if !__TBB_NO_IMPLICIT_LINKAGE
        #ifdef __TBB_LIB_NAME
	        #pragma comment(lib, __TBB_STRING(__TBB_LIB_NAME))
        #else
			#ifdef _DEBUG
				#pragma comment(lib, "tbb_debug.lib")
			#else
				#pragma comment(lib, "tbb.lib")
			#endif
        #endif
    #endif
#endif
