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

#ifndef __TBB_runtime_loader_H
#define __TBB_runtime_loader_H

#if ! TBB_PREVIEW_RUNTIME_LOADER
    #error Set TBB_PREVIEW_RUNTIME_LOADER to include runtime_loader.h
#endif

#include "tbb_stddef.h"
#include <climits>

#if _MSC_VER
    #if ! __TBB_NO_IMPLICIT_LINKAGE
        #ifdef _DEBUG
            #pragma comment( linker, "/nodefaultlib:tbb_debug.lib" )
            #pragma comment( linker, "/defaultlib:tbbproxy_debug.lib" )
        #else
            #pragma comment( linker, "/nodefaultlib:tbb.lib" )
            #pragma comment( linker, "/defaultlib:tbbproxy.lib" )
        #endif
    #endif
#endif

namespace tbb {

namespace interface6 {

//! Load TBB at runtime.
/*!

\b Usage:

In source code:

\code
#include "tbb/runtime_loader.h"

char const * path[] = { "<install dir>/lib/ia32", NULL };
tbb::runtime_loader loader( path );

// Now use TBB.
\endcode

Link with \c tbbproxy.lib (or \c libtbbproxy.a) instead of \c tbb.lib (\c libtbb.dylib,
\c libtbb.so).

TBB library will be loaded at runtime from \c <install dir>/lib/ia32 directory.

\b Attention:

All \c runtime_loader objects (in the same module, i.e. exe or dll) share some global state.
The most noticeable piece of global state is loaded TBB library.
There are some implications:

    -   Only one TBB library can be loaded per module.

    -   If one object has already loaded TBB library, another object will not load TBB.
        If the loaded TBB library is suitable for the second object, both will use TBB
        cooperatively, otherwise the second object will report an error.

    -   \c runtime_loader objects will not work (correctly) in parallel due to absence of
        synchronization.

*/

class runtime_loader : tbb::internal::no_copy {

    public:

        //! Error mode constants.
        enum error_mode {
            em_status,     //!< Save status of operation and continue.
            em_throw,      //!< Throw an exception of tbb::runtime_loader::error_code type.
            em_abort       //!< Print message to \c stderr and call \c abort().
        }; // error_mode

        //! Error codes.
        enum error_code {
            ec_ok,         //!< No errors.
            ec_bad_call,   //!< Invalid function call (e. g. load() called when TBB is already loaded).
            ec_bad_arg,    //!< Invalid argument passed.
            ec_bad_lib,    //!< Invalid library found (e. g. \c TBB_runtime_version symbol not found).
            ec_bad_ver,    //!< TBB found but version is not suitable.
            ec_no_lib      //!< No suitable TBB library found.
        }; // error_code

        //! Initialize object but do not load TBB.
        runtime_loader( error_mode mode = em_abort );

        //! Initialize object and load TBB.
        /*!
            See load() for details.

            If error mode is \c em_status, call status() to check whether TBB was loaded or not.
        */
        runtime_loader(
            char const * path[],                           //!< List of directories to search TBB in.
            int          min_ver = TBB_INTERFACE_VERSION,  //!< Minimal suitable version of TBB.
            int          max_ver = INT_MAX,                //!< Maximal suitable version of TBB.
            error_mode   mode    = em_abort                //!< Error mode for this object.
        );

        //! Destroy object.
        ~runtime_loader();

        //! Load TBB.
        /*!
            The method searches the directories specified in \c path[] array for the TBB library.
            When the library is found, it is loaded and its version is checked. If the version is
            not suitable, the library is unloaded, and the search continues.

            \b Note:

            For security reasons, avoid using relative directory names. For example, never load
            TBB from current (\c "."), parent (\c "..") or any other relative directory (like
            \c "lib" ). Use only absolute directory names (e. g. "/usr/local/lib").

            For the same security reasons, avoid using system default directories (\c "") on
            Windows. (See http://www.microsoft.com/technet/security/advisory/2269637.mspx for
            details.)

            Neglecting these rules may cause your program to execute 3-rd party malicious code.

            \b Errors:
                -   \c ec_bad_call - TBB already loaded by this object.
                -   \c ec_bad_arg - \p min_ver and/or \p max_ver negative or zero,
                    or \p min_ver > \p max_ver.
                -   \c ec_bad_ver - TBB of unsuitable version already loaded by another object.
                -   \c ec_no_lib - No suitable library found.
        */
        error_code
        load(
            char const * path[],                           //!< List of directories to search TBB in.
            int          min_ver = TBB_INTERFACE_VERSION,  //!< Minimal suitable version of TBB.
            int          max_ver = INT_MAX                 //!< Maximal suitable version of TBB.

        );


        //! Report status.
        /*!
            If error mode is \c em_status, the function returns status of the last operation.
        */
        error_code status();

    private:

        error_mode const my_mode;
        error_code       my_status;
        bool             my_loaded;

}; // class runtime_loader

} // namespace interface6

using interface6::runtime_loader;

} // namespace tbb

#endif /* __TBB_runtime_loader_H */

