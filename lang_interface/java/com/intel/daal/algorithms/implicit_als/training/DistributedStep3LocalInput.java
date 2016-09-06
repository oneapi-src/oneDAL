/* file: DistributedStep3LocalInput.java */
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
import com.intel.daal.algorithms.implicit_als.PartialModel;
import com.intel.daal.data_management.data.HomogenNumericTable;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.data_management.data.KeyValueDataCollection;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__IMPLICIT_ALS__TRAINING__DISTRIBUTEDSTEP3LOCALINPUT"></a>
 * @brief %Input objects for the implicit ALS training algorithm in the third step of the distributed processing mode
 */
public final class DistributedStep3LocalInput extends com.intel.daal.algorithms.Input {

    /** @private */
    static {
        System.loadLibrary("JavaAPI");
    }

    public DistributedStep3LocalInput(DaalContext context, long cAlgorithm, Precision prec, TrainingMethod method) {
        super(context);
        this.cObject = cGetInput(cAlgorithm, prec.getValue(), method.getValue());
    }

    /**
     * Sets an input partial model object for the implicit ALS training algorithm
     * in the third step of the distributed processing mode
     * @param id      Identifier of the input object
     * @param val     Value of the input object
     */
    public void set(PartialModelInputId id, PartialModel val) {
        if (id != PartialModelInputId.partialModel) {
            throw new IllegalArgumentException("Incorrect PartialModelInputId");
        }
        cSetPartialModel(this.cObject, id.getValue(), val.getCObject());
    }

    /**
     * Returns an input partial model object for the implicit ALS training algorithm
     * in the third step of the distributed processing mode
     * @param id      Identifier of the input object
     * @return        %Input object that corresponds to the given identifier
     */
    public PartialModel get(PartialModelInputId id) {
        if (id != PartialModelInputId.partialModel) {
            throw new IllegalArgumentException("Incorrect PartialModelInputId"); // error processing
        }
        return new PartialModel(getContext(), cGetPartialModel(this.cObject, id.getValue()));
    }

    /**
     * Sets an input numeric table object for the implicit ALS training algorithm
     * in the third step of the distributed processing mode
     * @param id      Identifier of the input object
     * @param val     Value of the input object
     */
    public void set(Step3LocalNumericTableInputId id, NumericTable val) {
        if (id != Step3LocalNumericTableInputId.offset) {
            throw new IllegalArgumentException("Incorrect Step3LocalNumericTableInputId");
        }
        cSetNumericTable(this.cObject, id.getValue(), val.getCObject());
    }

    /**
     * Returns an input numeric table object for the implicit ALS training algorithm
     * in the third step of the distributed processing mode
     * @param id      Identifier of the input object
     * @return        %Input object that corresponds to the given identifier
     */
    public NumericTable get(Step3LocalNumericTableInputId id) {
        if (id != Step3LocalNumericTableInputId.offset) {
            throw new IllegalArgumentException("Incorrect Step3LocalNumericTableInputId"); // error processing
        }
        return new HomogenNumericTable(getContext(), cGetNumericTable(this.cObject, id.getValue()));
    }

    /**
     * Sets an input key-value data collection object for the implicit ALS training algorithm
     * in the third step of the distributed processing mode
     * @param id      Identifier of the input object
     * @param val     Value of the input object
     */
    public void set(Step3LocalCollectionInputId id, KeyValueDataCollection val) {
        if (id != Step3LocalCollectionInputId.partialModelBlocksToNode) {
            throw new IllegalArgumentException("Incorrect Step3LocalCollectionInputId");
        }
        cSetDataCollection(this.cObject, id.getValue(), val.getCObject());
    }

    /**
     * Returns an input key-value data collection object for the implicit ALS training algorithm
     * in the third step of the distributed processing mode
     * @param id      Identifier of the input object
     * @return        %Input object that corresponds to the given identifier
     */
    public KeyValueDataCollection get(Step3LocalCollectionInputId id) {
        if (id != Step3LocalCollectionInputId.partialModelBlocksToNode) {
            throw new IllegalArgumentException("Incorrect Step3LocalCollectionInputId"); // error processing
        }
        return new KeyValueDataCollection(getContext(), cGetDataCollection(this.cObject, id.getValue()));
    }

    private native long cGetInput(long cAlgorithm, int prec, int method);

    private native void cSetPartialModel(long cObject, int id, long partialModelAddr);
    private native long cGetPartialModel(long cObject, int id);

    private native void cSetNumericTable(long cObject, int id, long numericTableAddr);
    private native long cGetNumericTable(long cObject, int id);

    private native void cSetDataCollection(long cObject, int id, long numericTableAddr);
    private native long cGetDataCollection(long cObject, int id);
}
