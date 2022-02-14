/* file: Result.java */
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
 * @ingroup correlation_distance
 * @{
 */
package com.intel.daal.algorithms.cordistance;

import com.intel.daal.utils.*;
import com.intel.daal.data_management.data.Factory;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__CORDISTANCE__RESULT"></a>
 * \brief Results obtained with the compute() method of the correlation distance algorithm in the batch processing mode
 */
public final class Result extends com.intel.daal.algorithms.Result {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs the result of the correlation distance algorithm
     * @param context   Context to manage the result of the correlation distance algorithm
     */
    public Result(DaalContext context) {
        super(context);
        this.cObject = cNewResult();
    }

    public Result(DaalContext context, long cObject) {
        super(context, cObject);
    }

    /**
     * Returns the result of the correlation distance algorithm
     * @param id   Identifier of the result
     * @return         %Result that corresponds to the given identifier
     */
    public NumericTable get(ResultId id) {
        if (id != ResultId.correlationDistance) {
            throw new IllegalArgumentException("id unsupported");
        }
        return (NumericTable)Factory.instance().createObject(getContext(), cGetResultTable(cObject, id.getValue()));
    }

    /**
     * Sets object to store the result of the correlation distance algorithm
     * @param id   Identifier of the result
     * @param val  Object to store the result that corresponds to the given identifier
     */
    public void set(ResultId id, NumericTable val) {
        if (id != ResultId.correlationDistance) {
            throw new IllegalArgumentException("id unsupported");
        }
        cSetResultTable(cObject, id.getValue(), val.getCObject());
    }

    private native long cNewResult();

    private native long cGetResultTable(long cObject, int id);

    private native void cSetResultTable(long cObject, int id, long ntAddr);
}
/** @} */
