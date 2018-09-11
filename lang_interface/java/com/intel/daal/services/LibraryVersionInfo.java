/* file: LibraryVersionInfo.java */
/*******************************************************************************
* Copyright 2014-2018 Intel Corporation.
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
 * @defgroup services Services
 * @brief Contains classes that implement service functionality, including error handling, memory allocation, and library version information
 * @{
 */
package com.intel.daal.services;

import com.intel.daal.utils.*;
/**
 * @defgroup library_version_info Extracting Version Information
 * @brief Provides information about the version of Intel(R) Data Analytics Acceleration Library.
 * @ingroup services
 * @{
 */
/**
 *  <a name="DAAL-CLASS-SERVICES__LIBRARYVERSIONINFO"></a>
 * @brief Provides information about the version of Intel(R) Data Analytics Acceleration Library.
 * <!-- \n<a href="DAAL-REF-LIBRARYVERSIONINFO-STRUCTURE">LibraryVersionInfo structure details and Optimization Notice</a> -->
 */
public class LibraryVersionInfo {
    protected native String cGetProductStatus(long x);

    protected native String cGetBuild(long x);

    protected native String cGetBuildRev(long x);

    protected native String cGetName(long x);

    protected native String cGetProcessor(long x);

    protected native int cGetMajorVersion(long x);

    protected native int cGetMinorVersion(long x);

    protected native int cGetUpdateVersion(long x);

    protected native long cInit();

    protected long cLibraryVersionInfo;

    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     *Default constructor
     */
    public LibraryVersionInfo() {
        cLibraryVersionInfo = cInit();
    }

    /**
     * Returns major library version number
     * @return Major library version
     */
    public int getMajorVersion() {
        return cGetMajorVersion(cLibraryVersionInfo);
    }

    /**
     * Returns minor library version number
     * @return Minor library version
     */
    public int getMinorVersion() {
        return cGetMinorVersion(cLibraryVersionInfo);
    }

    /**
     * Returns update library version number
     * @return Update library version
     */
    public int getUpdateVersion() {
        return cGetUpdateVersion(cLibraryVersionInfo);
    }

    /**
     * Returns product library status (alfa/beta/product)
     * @return Product library status
     */
    public String getProductStatus() {
        return cGetProductStatus(cLibraryVersionInfo);
    }

    /**
     * Returns library build
     * @return Library build
     */
    public String getBuild() {
        return cGetBuild(cLibraryVersionInfo);
    }

    /**
     * Returns library build revision
     * @return Library build revision
     */
    public String getBuildRev() {
        return cGetBuildRev(cLibraryVersionInfo);
    }

    /**
     * Returns library name
     * @return Library name
     */
    public String getName() {
        return cGetName(cLibraryVersionInfo);
    }

    /**
    * Returns the instruction set supported by the processor
     * @return the instruction set supported by the processor
     */
    public String getProcessor() {
        return cGetProcessor(cLibraryVersionInfo);
    }
}
/** @} */
/** @} */
