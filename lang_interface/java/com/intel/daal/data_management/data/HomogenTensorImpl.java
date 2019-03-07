/* file: HomogenTensorImpl.java */
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
 * @ingroup tensor
 * @{
 */
package com.intel.daal.data_management.data;

import com.intel.daal.utils.*;
import com.intel.daal.services.DaalContext;

/**
 * @brief A derivative class of the TensorImpl class, that provides common interfaces for
 *        different implementations of a homogen tensor
 */
abstract class HomogenTensorImpl extends TensorImpl {
    protected Class<? extends Number> type;

    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs the homogen tensor
     * @param context   Context to manage the homogen tensor
     */
    public HomogenTensorImpl(DaalContext context) {
        super(context);
    }

    abstract public Object getDataObject();

    abstract public Class<? extends Number> getNumericType();
}
/** @} */
