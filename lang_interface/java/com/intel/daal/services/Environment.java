/* file: Environment.java */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation.
*
* This software and the related documents are Intel copyrighted  materials,  and
* your use of  them is  governed by the  express license  under which  they were
* provided to you (License).  Unless the License provides otherwise, you may not
* use, modify, copy, publish, distribute,  disclose or transmit this software or
* the related documents without Intel's prior written permission.
*
* This software and the related documents  are provided as  is,  with no express
* or implied  warranties,  other  than those  that are  expressly stated  in the
* License.
*******************************************************************************/

/**
 * @defgroup env_detect Managing the Computational Environment
 * @brief Provides methods to interact with the environment, including processor detection and control by the number of threads.
 * @ingroup services
 * @{
 */
package com.intel.daal.services;

import com.intel.daal.utils.*;
/**
 *  <a name="DAAL-CLASS-SERVICES__ENVIRONMENT"></a>
 * @brief Provides information about computational environment.
 */
public class Environment {
    protected static native int  cGetCpuId(int enable);
    protected static native int  cSetCpuId(int cpuid);
    protected static native int  cEnableInstructionsSet(int enable);
    protected static native void cSetNumberOfThreads(int numThreads);
    protected static native int  cGetNumberOfThreads();
    protected static native void cEnableThreadPinning(boolean enableThreadPinningFlag);

    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
    *  Detects the processor type
    *  \param[in] enable  An enabling flag
    *  \return The CPU ID
    */
    public static int getCpuId(CpuTypeEnable enable) {
        return cGetCpuId(enable.getValue());
    }

    /**
    *  Set the processor type
    *  \param[in] cpuid  CPU ID
    *  \return  CPU ID if success; -1 if error
    */
    public static int setCpuId(CpuType cpuid) {
        return cSetCpuId(cpuid.getValue());
    }

    /**
    *  Set
    *  \param[in] enable  An enabling flag
    *  \return  CPU ID
    */
    public static int enableInstructionsSet(CpuTypeEnable enable) {
        return cEnableInstructionsSet(enable.getValue());
    }

    /**
     * Sets the number of threads to be used in the application
     * @param numThreads  The number of threads to set
     */
    public static void setNumberOfThreads(int numThreads) {
        cSetNumberOfThreads(numThreads);
    }

    /**
     * Enables thread pinning
     * @param enableThreadPinningFlag  Flag to thread pinning enable
     */
    public static void enableThreadPinning(boolean enableThreadPinningFlag) {
        cEnableThreadPinning(enableThreadPinningFlag);
    }

    /**
    * Returns number of threads used by the application
    * @return Number of threads
    */
    public static int getNumberOfThreads() {
        return cGetNumberOfThreads();
    }
}
/** @} */
