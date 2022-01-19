/* file: Input.java */
/*******************************************************************************
* Copyright 2014-2022 Intel Corporation
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
 * @ingroup svd
 * @{
 */
package com.intel.daal.algorithms.svd;

import com.intel.daal.utils.*;
import com.intel.daal.data_management.data.Factory;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__SVD__INPUT"></a>
 * @brief %Input objects for the SVD algorithm in the batch processing and online processing modes, and the first step in the
 * distributed processing mode
 */
public class Input extends com.intel.daal.algorithms.Input {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    public Input(DaalContext context, long cObject) {
        super(context, cObject);
    }

    /**
     * Sets an input object for the SVD algorithm
     * @param id    Identifier of the %input object
     * @param val   Value of the input object
     */
    public void set(InputId id, NumericTable val) {
        if (id == InputId.data) {
            cSetInputTable(cObject, id.getValue(), val.getCObject());
        } else {
            throw new IllegalArgumentException("id unsupported");
        }
    }

    /**
     * Returns an input object for the SVD algorithm
     * @param id Identifier of the %input object
     * @return   %Input object that corresponds to the given identifier
     */
    public NumericTable get(InputId id) {
        if (id == InputId.data) {
            return (NumericTable)Factory.instance().createObject(getContext(), cGetInputTable(cObject, id.getValue()));
        } else {
            throw new IllegalArgumentException("id unsupported");
        }
    }

    private native void cSetInputTable(long cInput, int id, long ntAddr);

    private native long cGetInputTable(long cInput, int id);
}
/** @} */
