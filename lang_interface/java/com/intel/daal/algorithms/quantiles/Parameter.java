/* file: Parameter.java */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation
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
