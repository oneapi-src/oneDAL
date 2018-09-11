/* file: Parameter.java */
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
 * @ingroup quantiles
 * @{
 */
package com.intel.daal.algorithms.quantiles;

import com.intel.daal.utils.*;
import com.intel.daal.data_management.data.Factory;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__QUANTILES__PARAMETER"></a>
 * @brief Parameters of the quantiles algorithm
 */
public class Parameter extends com.intel.daal.algorithms.Parameter {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs the parameter of the quality metric algorithm
     * @param context   Context to manage the parameter of the quality metric algorithm
     */
    public Parameter(DaalContext context) {
        super(context);
    }

    public Parameter(DaalContext context, long cObject) {
        super(context, cObject);
    }

    /**
     * Sets the parameter of the quantiles algorithm
     * @param quantiles Identifier of the parameter
     */
    public void setQuantileOrders(NumericTable quantiles) {
        cSetQuantileOrders(this.cObject, quantiles.getCObject());
    }

    /**
     * Gets the parameter of the quantiles algorithm
     * @return    Identifier of the parameter
     */
    public NumericTable getQuantileOrders() {
        return (NumericTable)Factory.instance().createObject(getContext(), cGetQuantileOrders(this.cObject));
    }

    private native void cSetQuantileOrders(long parAddr, long quantilesAddr);

    private native long cGetQuantileOrders(long parAddr);
}
/** @} */
