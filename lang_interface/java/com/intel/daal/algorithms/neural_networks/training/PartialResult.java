/* file: PartialResult.java */
/*******************************************************************************
* Copyright 2014-2017 Intel Corporation
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
 * @defgroup neural_networks_training Training
 * @brief Contains classes for training the model of the neural network
 * @ingroup neural_networks
 * @{
 */
package com.intel.daal.algorithms.neural_networks.training;

import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.algorithms.ComputeStep;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.data_management.data.Factory;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__TRAINING__PARTIALRESULT"></a>
 * @brief Provides methods to access partial results obtained with the compute() method of the
 *        neural network training algorithm in the distributed processing mode
 */
public class PartialResult extends com.intel.daal.algorithms.PartialResult {
    /** @private */
    static {
        System.loadLibrary("JavaAPI");
    }

    /**
     * Default constructor. Constructs empty PartialResult
     * @param context      Context to manage the partial result for the neural network training algorithm
     */
    public PartialResult(DaalContext context) {
        super(context);
        this.cObject = cNewPartialResult();
    }

    public PartialResult(DaalContext context, long cPartialResult) {
        super(context, cPartialResult);
    }

    /**
     * Returns a partial result of the neural network training algorithm
     * @param  id   Identifier of the partial result, @ref PartialResultId
     * @return Partial result that corresponds to the given identifier
     */
    public NumericTable get(PartialResultId id) {
        int idValue = id.getValue();
        if (idValue != PartialResultId.derivatives.getValue() && idValue != PartialResultId.batchSize.getValue()) {
            throw new IllegalArgumentException("id unsupported");
        }
        return (NumericTable)Factory.instance().createObject(getContext(), cGetPartialResultTable(getCObject(), idValue));
    }

    /**
     * Sets a partial result of the neural network training algorithm
     * @param id                 Identifier of the partial result, @ref PartialResultId
     * @param value              Value of the partial result
     */
    public void set(PartialResultId id, NumericTable value) {
        int idValue = id.getValue();
        if (idValue != PartialResultId.derivatives.getValue() && idValue != PartialResultId.batchSize.getValue()) {
            throw new IllegalArgumentException("id unsupported");
        }
        cSetPartialResultTable(getCObject(), idValue, value.getCObject());
    }

    private native long cNewPartialResult();
    private native long cGetPartialResultTable(long cPartialResult, int id);
    private native void cSetPartialResultTable(long cPartialResult, int id, long cNumericTable);
}
/** @} */
