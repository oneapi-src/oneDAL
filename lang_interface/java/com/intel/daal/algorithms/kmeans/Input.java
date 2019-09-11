/* file: Input.java */
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
 * @ingroup kmeans_compute
 * @{
 */
package com.intel.daal.algorithms.kmeans;

import com.intel.daal.utils.*;
import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.data_management.data.Factory;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__KMEANS__INPUT"></a>
 * @brief  %Input objects for the K-Means algorithm
 */
public class Input extends com.intel.daal.algorithms.Input {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    public Input(DaalContext context, long cObject) {
        super(context, cObject);
        this.cObject = cObject;
    }

    /**
     * Sets an input object for the K-Means algorithm
     * @param id   Identifier of the input object
     * @param val  Value of the input object
     */
    public void set(InputId id, NumericTable val) {
        if (id == InputId.data || id == InputId.inputCentroids) {
            cSetData(cObject, id.getValue(), val.getCObject());
        } else {
            throw new IllegalArgumentException("id unsupported");
        }
    }

    /**
     * Returns an input object for the K-Means algorithm
     * @param id Identifier of the input object
     * @return   %Input object that corresponds to the given identifier
     */
    public NumericTable get(InputId id) {
        if (id == InputId.data || id == InputId.inputCentroids) {
            return (NumericTable)Factory.instance().createObject(getContext(), cGetData(cObject, id.getValue()));
        } else {
            throw new IllegalArgumentException("id unsupported");
        }
    }

    private native void cSetData(long inputAddr, int id, long ntAddr);

    private native long cGetData(long inputAddr, int id);
}
/** @} */
