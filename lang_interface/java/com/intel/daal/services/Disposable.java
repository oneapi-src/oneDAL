/* file: Disposable.java */
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
 * @ingroup memory
 * @{
 */
package com.intel.daal.services;

/**
 *  <a name="DAAL-CLASS-SERVICES__DISPOSABLE"></a>
 * @brief Class that frees memory allocated for the native C++ object
 */
public interface Disposable {
    /**
     * Releases memory allocated for the native object
     */
    abstract void dispose();
}
/** @} */
