/* file: InitInput.java */
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
 * @ingroup kmeans_init
 * @{
 */
package com.intel.daal.algorithms.kmeans.init;

import com.intel.daal.utils.*;
import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.data_management.data.Factory;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.services.DaalContext;
import com.intel.daal.data_management.data.Factory;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__KMEANS__INIT__INITINPUT"></a>
 * @brief  %InitInput objects for computing initial clusters for the K-Means algorithm
 */
public class InitInput extends com.intel.daal.algorithms.Input {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    public InitInput(DaalContext context, long cObject) {
        super(context, cObject);
    }

    /**
     * Sets an input object for computing initial clusters for the K-Means algorithm
     * @param id   Identifier of the input object
     * @param val  Value of the input object     */
    public void set(InitInputId id, NumericTable val) {
        if (id == InitInputId.data) {
            cSetData(cObject, id.getValue(), val.getCObject());
        } else {
            throw new IllegalArgumentException("id unsupported");
        }
    }

    /**
     * Returns an input object for computing initial clusters for the K-Means algorithm
     * @param id Identifier of the input object
     * @return   Input object that corresponds to the given identifier
     */
    public NumericTable get(InitInputId id) {
        if (id == InitInputId.data) {
            return (NumericTable)Factory.instance().createObject(getContext(), cGetData(cObject, id.getValue()));
        } else {
            throw new IllegalArgumentException("id unsupported");
        }
    }

    private native void cSetData(long inputAddr, int id, long ntAddr);

    private native long cGetData(long inputAddr, int id);
}
/** @} */
