/* file: Input.java */
/*******************************************************************************
* Copyright 2014 Intel Corporation
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
 * @ingroup saga
 * @{
 */
package com.intel.daal.algorithms.optimization_solver.saga;

import com.intel.daal.utils.*;
import com.intel.daal.data_management.data.Factory;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__OPTIMIZATION_SOLVER__SAGA__INPUT"></a>
 * @brief %Input objects for the SAGA algorithm
 */
public class Input extends com.intel.daal.algorithms.optimization_solver.iterative_solver.Input {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs the input for the SAGA algorithm
     * @param context       Context to manage the input for the SAGA algorithm
     */
    public Input(DaalContext context) {
        super(context);
    }

    /**
     * Constructs the input for the SAGA algorithm
     * @param context       Context to manage the SAGA algorithm
     * @param cInput        Pointer to C++ implementation of the input
     */
    public Input(DaalContext context, long cInput) {
        super(context, cInput);
    }

    /**
     * Sets an optional input data for the SAGA algorithm
     * @param id    Identifier of the optional data object
     * @param val   The optional data object
     */
    public void set(OptionalDataId id, NumericTable val) {
        if (id != OptionalDataId.gradientsTable) {
            throw new IllegalArgumentException("id unsupported");
        }
        cSetOptionalData(cObject, id.getValue(), val.getCObject());
    }

    /**
     * Returns an optional input data for the SAGA algorithm
     * @param id Identifier of the optional data object
     * @return   %Optional data object that corresponds to the given identifier
     */
    public NumericTable get(OptionalDataId id) {
        if (id != OptionalDataId.gradientsTable) {
            throw new IllegalArgumentException("id unsupported");
        }
        return (NumericTable)Factory.instance().createObject(getContext(), cGetOptionalData(cObject, id.getValue()));
    }

    protected native void cSetOptionalData(long cInput, int id, long ntAddr);
    protected native long cGetOptionalData(long cInput, int id);
}
/** @} */
