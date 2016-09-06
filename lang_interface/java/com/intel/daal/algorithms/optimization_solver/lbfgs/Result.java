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

package com.intel.daal.algorithms.optimization_solver.lbfgs;

import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.algorithms.ComputeStep;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.algorithms.OptionalArgument;
import com.intel.daal.data_management.data.HomogenNumericTable;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.services.DaalContext;
import com.intel.daal.data_management.data.Factory;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__OPTIMIZATION_SOLVER__LBFGS__RESULT"></a>
 * @brief Provides methods to access the results obtained with the compute() method of the
 *        iterative algorithm in the batch processing mode
 */
public class Result extends com.intel.daal.algorithms.optimization_solver.iterative_solver.Result {
    /** @private */
    static {
        System.loadLibrary("JavaAPI");
    }

    /**
     * Constructs the result for the iterative algorithm
     * @param context Context to manage objective function algorithm
     */
    public Result(DaalContext context) {
        super(context);
        this.cObject = cNewResult();
    }

    /**
    * Constructs the result for the iterative algorithm
    * @param context       Context to manage the iterative algorithm result
    * @param cResult       Pointer to C++ implementation of the result
    */
    public Result(DaalContext context, long cResult) {
        super(context, cResult);
    }

    /**
     * Returns the optional data of the iterative algorithm in the batch processing mode
     * @param id   Identifier of the optional data, @ref OptionalDataId
     * @return Optional data that corresponds to the given identifier
     */
    public NumericTable get(OptionalDataId id) {
        if (id != OptionalDataId.correctionPairs &&
            id != OptionalDataId.correctionIndices &&
            id != OptionalDataId.averageArgumentLIterations) {
            throw new IllegalArgumentException("id unsupported");
        }
        return (NumericTable)Factory.instance().createObject(getContext(), cGetOptionalData(cObject, id.getValue()));
    }

    /**
     * Sets the optional data of the iterative algorithm in the batch processing mode
     * @param id   Identifier of the optional data, @ref OptionalDataId
     * @param val Object to store the data
     */
    public void set(OptionalDataId id, NumericTable val) {
        if (id != OptionalDataId.correctionPairs &&
            id != OptionalDataId.correctionIndices &&
            id != OptionalDataId.averageArgumentLIterations) {
            throw new IllegalArgumentException("id unsupported");
        }
        cSetOptionalData(cObject, id.getValue(), val.getCObject());
    }

    private native long cNewResult();
    protected native long cGetOptionalData(long cObject, int id);
    protected native void cSetOptionalData(long cResult, int id, long cOptionalData);
}
