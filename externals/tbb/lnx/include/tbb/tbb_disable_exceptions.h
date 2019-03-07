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

//! To disable use of exceptions, include this header before any other header file from the library.

//! The macro that prevents use of exceptions in the library files
#undef  TBB_USE_EXCEPTIONS
#define TBB_USE_EXCEPTIONS 0

//! Prevent compilers from issuing exception related warnings.
/** Note that the warnings are suppressed for all the code after this header is included. */
#if _MSC_VER
#if __INTEL_COMPILER
    #pragma warning (disable: 583)
#else
    #pragma warning (disable: 4530 4577)
#endif
#endif
