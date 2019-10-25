/* file: InitDistributedStep1LocalInput.java */
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
 * @ingroup dbscan_distributed
 * @{
 */
package com.intel.daal.algorithms.gbt.regression.init;

import com.intel.daal.utils.*;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.data_management.data.Factory;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__DBSCAN__InitDistributedStep1LocalInput"></a>
 * @brief %Input objects for the DBSCAN algorithm in the first step of the distributed processing mode
 */

public final class InitDistributedStep1LocalInput extends com.intel.daal.algorithms.Input {

    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    public InitDistributedStep1LocalInput(DaalContext context, long cObject) {
        super(context);
        this.cObject = cObject;
    }

    /**
     * Sets an input object for the DBSCAN algorithm in the first step of the distributed processing mode
     * @param id      Identifier of the input object
     * @param val     Value of the input object
     */
    public void set(InitStep1LocalNumericTableInputId id, NumericTable val) {
        if (id == InitStep1LocalNumericTableInputId.step1LocalData || id == InitStep1LocalNumericTableInputId.step1LocalDependentVariables) {
            cSetNumericTable(this.cObject, id.getValue(), val.getCObject());
        } else {
            throw new IllegalArgumentException("id unsupported");
        }
    }

    /**
     * Returns an input object for the DBSCAN algorithm in the first step of the distributed processing mode
     * @param id      Identifier of the input object
     * @return        %Input object that corresponds to the given identifier
     */
    public NumericTable get(InitStep1LocalNumericTableInputId id) {
        if (id == InitStep1LocalNumericTableInputId.step1LocalData || id == InitStep1LocalNumericTableInputId.step1LocalDependentVariables) {
            return (NumericTable)Factory.instance().createObject(getContext(), cGetNumericTable(getCObject(), id.getValue()));
        } else {
            throw new IllegalArgumentException("id unsupported");
        }
    }

    private native void cSetNumericTable(long cObject, int id, long ntAddr);
    private native long cGetNumericTable(long cObject, int id);
}
/** @} */
