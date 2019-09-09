/* file: TransformResult.java */
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
 * @ingroup pca_transform
 * @{
 */
package com.intel.daal.algorithms.pca.transform;

import com.intel.daal.utils.*;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.data_management.data.Factory;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__PCA__TRANSFORM__RESULT"></a>
 * @brief Results obtained with the compute() method of the PCA transformation algorithm in the batch processing mode
 */
public final class TransformResult extends com.intel.daal.algorithms.Result {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs the result of PCA transformation algorithm
     * @param context   Context to manage the result of PCA transformation algorithm
     */
    public TransformResult(DaalContext context) {
        super(context);
        this.cObject = cNewResult();
    }

    public TransformResult(DaalContext context, long cObject) {
        super(context, cObject);
    }

    /**
     * Returns the result of PCA transformation
     * @param  id   Identifier of the result
     * @return Result that corresponds to the given identifier
     */
    public NumericTable get(TransformResultId id) {
        int idValue = id.getValue();
        if (idValue != TransformResultId.transformedData.getValue()) {
            throw new IllegalArgumentException("id unsupported");
        }
        return (NumericTable)Factory.instance().createObject(getContext(), cGetTransformedData(cObject));
    }

    /**
     * Sets the final result of the PCA transformation algorithm
     * @param id   Identifier of the result
     * @param val  Result that corresponds to the given identifier
     */
    public void set(TransformResultId id, NumericTable val) {
        int idValue = id.getValue();
        if (idValue != TransformResultId.transformedData.getValue()) {
            throw new IllegalArgumentException("id unsupported");
        }
        cSetTransformedData(cObject, val.getCObject());
    }

    private native long cNewResult();
    private native long cGetTransformedData(long cObject);
    private native void cSetTransformedData(long cObject, long cNumericTable);
}
/** @} */
