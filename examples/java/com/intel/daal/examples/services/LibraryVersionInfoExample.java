/* file: LibraryVersionInfoExample.java */
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

        System.out.println("Major Version:  " + vi.getMajorVersion());
        System.out.println("Minor Version:  " + vi.getMinorVersion());
        System.out.println("Update Version: " + vi.getUpdateVersion());
        System.out.println("Product:        " + vi.getProductStatus());
        System.out.println("Build:          " + vi.getBuild());
        System.out.println("Name:           " + vi.getName());
        System.out.println("Processor:      " + vi.getProcessor());

        System.out.println("CPU:            " + Environment.getCpuId(CpuTypeEnable.cpu_default));
    }
}
