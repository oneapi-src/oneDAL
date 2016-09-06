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

package com.intel.daal.algorithms.optimization_solver.objective_function;

import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.algorithms.ComputeStep;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.data_management.data.HomogenNumericTable;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.data_management.data.DataCollection;
import com.intel.daal.services.DaalContext;
import com.intel.daal.data_management.data.Factory;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__OPTIMIZATION_SOLVER__OBJECTIVE_FUNCTION__RESULT"></a>
 * @brief Provides methods to access the results obtained with the compute() method of the
 *        objective funtion algorithm in the batch processing mode
 */
public final class Result extends com.intel.daal.algorithms.Result {
    /** @private */
    static {
        System.loadLibrary("JavaAPI");
    }

    /**
     * Constructs the result for objective function algorithm
     * @param context       Context to manage objective function algorithm
     */
    public Result(DaalContext context) {
        super(context);
        this.cObject = cNewResult();
    }

    /**
     * Constructs the result for objective function algorithm
     * @param context       Context to manage objective function algorithm result
     * @param cResult       Pointer to C++ implementation of the result
     */
    public Result(DaalContext context, long cResult) {
        super(context, cResult);
    }

    /**
     * Returns the result of the objective funtion algorithm in the batch processing mode
     * @param id Identifier of the result, @ref ResultId
     * @return Result that corresponds to the given identifier
     */
    public DataCollection get(ResultId id) {
        if (id != ResultId.resultCollection) {
            throw new IllegalArgumentException("id unsupported");
        }
        return new DataCollection(getContext(), cGetResultDataCollection(cObject, id.getValue()));
    }

    /**
     * Returns the result of the objective funtion algorithm in the batch processing mode
     * @param id Identifier of the result, @ref ResultId
     * @param index Identifier of result table index in the resul collection, @ref ResultCollectionId
     * @return Result that corresponds to the given identifier
     */
    public NumericTable get(ResultId id, ResultCollectionId index) {
        if (id != ResultId.resultCollection) {
            throw new IllegalArgumentException("id unsupported");
        }
        if(index != ResultCollectionId.gradientIdx &&
           index != ResultCollectionId.valueIdx &&
           index != ResultCollectionId.hessianIdx) {
            throw new IllegalArgumentException("index argument for this id unsupported");
        }
        return (NumericTable)Factory.instance().createObject(getContext(), cGetResultTable(cObject, id.getValue(), index.getValue()));
    }

    /**
     * Sets the result of the objective funtion algorithm in the batch processing mode
     * @param id Identifier of the result, @ref ResultId
     * @param val Object to store the result
     */
    public void set(ResultId id, DataCollection val) {
        if (id != ResultId.resultCollection) {
            throw new IllegalArgumentException("id unsupported");
        }
        cSetResultDataCollection(cObject, id.getValue(), val.getCObject());
    }

    private native long cNewResult();

    private native long cGetResultDataCollection(long cObject, int id);

    private native void cSetResultDataCollection(long cResult, int id, long cNumericTable);

    private native long cGetResultTable(long cObject, int id, int index);
}
