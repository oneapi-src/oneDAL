/* file: DistributedStep4LocalInput.java */
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

package com.intel.daal.algorithms.implicit_als.training;

import com.intel.daal.algorithms.Precision;
import com.intel.daal.data_management.data.HomogenNumericTable;
import com.intel.daal.data_management.data.CSRNumericTable;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.data_management.data.KeyValueDataCollection;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__IMPLICIT_ALS__TRAINING__DISTRIBUTEDSTEP4LOCALINPUT"></a>
 * @brief %Input objects for the implicit ALS training algorithm in the fourth step of the distributed processing mode
 */
public final class DistributedStep4LocalInput extends com.intel.daal.algorithms.Input {

    /** @private */
    static {
        System.loadLibrary("JavaAPI");
    }

    public DistributedStep4LocalInput(DaalContext context, long cAlgorithm, Precision prec, TrainingMethod method) {
        super(context);
        this.cObject = cGetInput(cAlgorithm, prec.getValue(), method.getValue());
    }

    /**
     * Sets an input key-value data collection object for the implicit ALS training algorithm
     * in the fourth step of the distributed processing mode
     * @param id      Identifier of the input object
     * @param val     Value of the input object
     */
    public void set(Step4LocalPartialModelsInputId id, KeyValueDataCollection val) {
        if (id != Step4LocalPartialModelsInputId.partialModels) {
            throw new IllegalArgumentException("Incorrect Step4LocalPartialModelsInputId");
        }
        cSetDataCollection(this.cObject, id.getValue(), val.getCObject());
    }

    /**
     * Returns an input key-value data collection object for the implicit ALS training algorithm
     * in the fourth step of the distributed processing mode
     * @param id      Identifier of the input object
     * @return        %Input object that corresponds to the given identifier
     */
    public KeyValueDataCollection get(Step4LocalPartialModelsInputId id) {
        if (id != Step4LocalPartialModelsInputId.partialModels) {
            throw new IllegalArgumentException("Incorrect Step4LocalPartialModelsInputId"); // error processing
        }
        return new KeyValueDataCollection(getContext(), cGetDataCollection(this.cObject, id.getValue()));
    }

    /**
     * Sets an input numeric table object for the implicit ALS training algorithm
     * in the fourth step of the distributed processing mode
     * @param id      Identifier of the input object
     * @param val     Value of the input object
     */
    public void set(Step4LocalNumericTableInputId id, NumericTable val) {
        if (id != Step4LocalNumericTableInputId.partialData &&
            id != Step4LocalNumericTableInputId.inputOfStep4FromStep2) {
            throw new IllegalArgumentException("Incorrect Step4LocalNumericTableInputId");
        }
        cSetNumericTable(this.cObject, id.getValue(), val.getCObject());
    }

    /**
     * Returns an input numeric table object for the implicit ALS training algorithm
     * in the fourth step of the distributed processing mode
     * @param id      Identifier of the input object
     * @return        %Input object that corresponds to the given identifier
     */
    public NumericTable get(Step4LocalNumericTableInputId id) {
        if (id != Step4LocalNumericTableInputId.partialData &&
            id != Step4LocalNumericTableInputId.inputOfStep4FromStep2) {
            throw new IllegalArgumentException("Incorrect Step4LocalNumericTableInputId"); // error processing
        }
        if (id == Step4LocalNumericTableInputId.partialData) {
            return new CSRNumericTable(getContext(), cGetNumericTable(this.cObject, id.getValue()));
        } else /* if (id == Step4LocalNumericTableInputId.inputOfStep4FromStep2) */ {
            return new HomogenNumericTable(getContext(), cGetNumericTable(this.cObject, id.getValue()));
        }
    }

    private native long cGetInput(long cAlgorithm, int prec, int method);

    private native void cSetNumericTable(long cObject, int id, long numericTableAddr);
    private native long cGetNumericTable(long cObject, int id);

    private native void cSetDataCollection(long cObject, int id, long collectionAddr);
    private native long cGetDataCollection(long cObject, int id);
}
