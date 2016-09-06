/* file: Parameter.java */
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

package com.intel.daal.algorithms.low_order_moments;

import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__LOW_ORDER_MOMENTS__PARAMETER"></a>
 * @brief Parameters of the low order %moments algorithm
 */
public class Parameter extends com.intel.daal.algorithms.Parameter {
    /** @private */
    static {
        System.loadLibrary("JavaAPI");
    }

    public Parameter(DaalContext context, long cObject) {
        super(context, cObject);
        _initializationProcedure = null;
    }

    /**
     * Sets the initialization procedure
     * @param initializationProcedure   Initialization procedure
     */
    public void setInitializationProcedure(InitializationProcedureIface initializationProcedure) {
        _initializationProcedure = initializationProcedure;
        cSetInitializationProcedure(this.cObject, _initializationProcedure.getCObject());
    }

    /**
     * Gets initialization procedure
     * @return  Initialization procedure
     */
    public InitializationProcedureIface getInitializationProcedure() {
        if (_initializationProcedure == null) {
            _initializationProcedure = new DefaultInitializationProcedure();
        }
        return _initializationProcedure;
    }

    private InitializationProcedureIface _initializationProcedure;

    private native void cSetInitializationProcedure(long cParameter, long cInitProcedure);
}
