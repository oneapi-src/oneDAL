/* file: ErrorHandling.java */
/*******************************************************************************
* Copyright 2020-2021 Intel Corporation
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

/**
 * @brief Auxiliary function used in exception handling in Java
 * @ingroup services
 * @{
 */
package com.intel.daal.services;

import java.lang.System;
import java.lang.Throwable;
/**
 *  <a name="DAAL-CLASS-SERVICES__ERRORHANDLING"></a>
 * @brief Provides auxiliary function used in exception handling in Java.
 */
public class ErrorHandling {
    /**
    *  Print information from a Throwable object
    *  @param e  Throwable object
    */
    public static void printThrowable(Throwable e) {
        System.out.println("Error: exception caught: " + e);
        System.out.println("Exception message: " + e.getMessage());
        System.out.println("Exception cause: " + e.getCause());
        e.printStackTrace();
    }
}
/** @} */
