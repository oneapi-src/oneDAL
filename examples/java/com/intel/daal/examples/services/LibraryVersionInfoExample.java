/* file: LibraryVersionInfoExample.java */
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

/*
//  Content:
//  Intel(R) DAAL version information
////////////////////////////////////////////////////////////////////////////////
*/

/**
 * <a name="DAAL-EXAMPLE-JAVA-LIBRARYVERSIONINFOEXAMPLE">
 * @example LibraryVersionInfoExample.java
 */

package com.intel.daal.examples.services;

import com.intel.daal.services.CpuTypeEnable;
import com.intel.daal.services.Environment;
import com.intel.daal.services.LibraryVersionInfo;

class LibraryVersionInfoExample {
    public static void main(String[] args) throws java.io.FileNotFoundException, java.io.IOException {
        LibraryVersionInfo vi = new LibraryVersionInfo();

        System.out.println("Major version:          " + vi.getMajorVersion());
        System.out.println("Minor version:          " + vi.getMinorVersion());
        System.out.println("Update version:         " + vi.getUpdateVersion());
        System.out.println("Product status:         " + vi.getProductStatus());
        System.out.println("Build:                  " + vi.getBuild());
        System.out.println("Build revision:         " + vi.getBuildRev());
        System.out.println("Name:                   " + vi.getName());
        System.out.println("Processor optimization: " + vi.getProcessor());
    }
}
