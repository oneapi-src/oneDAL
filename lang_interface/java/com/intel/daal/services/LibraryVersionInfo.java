/* file: LibraryVersionInfo.java */
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

package com.intel.daal.services;

/**
 *  <a name="DAAL-CLASS-SERVICES__LIBRARYVERSIONINFO"></a>
 * @brief Provides information about the version of Intel(R) Data Analytics Acceleration Library.
 * \n<a href="DAAL-REF-LIBRARYVERSIONINFO-STRUCTURE">LibraryVersionInfo structure details and Optimization Notice</a>
 */
public class LibraryVersionInfo {
    protected native String cGetProductStatus(long x);

    protected native String cGetBuild(long x);

    protected native String cGetName(long x);

    protected native String cGetProcessor(long x);

    protected native int cGetMajorVersion(long x);

    protected native int cGetMinorVersion(long x);

    protected native int cGetUpdateVersion(long x);

    protected native long cInit();

    protected long cLibraryVersionInfo;

    /** @private */
    static {
        System.loadLibrary("JavaAPI");
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
