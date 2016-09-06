/* file: Result.java */
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

package com.intel.daal.algorithms.pca;

import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.algorithms.ComputeStep;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.data_management.data.HomogenNumericTable;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__PCA__RESULT"></a>
 * @brief Provides methods to access final results obtained with the compute() method of PCA algorithm in the batch
 *        processing mode, or finalizeCompute() method in the online or distributed processing mode
 */
public class Result extends com.intel.daal.algorithms.Result {
    /** @private */
    static {
        System.loadLibrary("JavaAPI");
    }

    /**
    * Default constructor
    * @param context   Context to manage the PCA algorithm result
    */
    public Result(DaalContext context) {
        super(context);
        this.cObject = cNewResult();
    }

    public Result(DaalContext context, long cObject) {
        super(context, cObject);
    }

    /**
     * Returns final result of the PCA algorithm
     * @param  id   Identifier of the result, @ref ResultId
     * @return Final result that corresponds to the given identifier
     */
    public NumericTable get(ResultId id) {
        if (id != ResultId.eigenValues && id != ResultId.eigenVectors) {
            throw new IllegalArgumentException("id unsupported");
        }
        int idValue = id.getValue();
        return new HomogenNumericTable(getContext(), cGetResultTable(cObject, idValue));
    }

    /**
     * Sets final result of the PCA algorithm
     * @param id    Identifier of the final result
     * @param value Object to store final result
     */
    public void set(ResultId id, NumericTable value) {
        if (id != ResultId.eigenValues && id != ResultId.eigenVectors) {
            throw new IllegalArgumentException("id unsupported");
        }
        int idValue = id.getValue();
        cSetResultTable(cObject, idValue, value.getCObject());
    }

    private native long cNewResult();

    private native long cGetResultTable(long cResult, int id);

    private native void cSetResultTable(long cResult, int id, long cNumericTable);
}
