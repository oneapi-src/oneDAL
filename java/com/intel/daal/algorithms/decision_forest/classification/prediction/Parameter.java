/* file: Parameter.java */
/*******************************************************************************
* Copyright 2014-2021 Intel Corporation
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
 * @ingroup decision_forest_classification_prediction
 */
/**
 * @brief Contains parameter class for decision forest algorithm
 */
package com.intel.daal.algorithms.decision_forest.classification.prediction;

import com.intel.daal.services.DaalContext;
import com.intel.daal.algorithms.decision_forest.classification.prediction.VotingMethod;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__DECISION_FOREST__CLASSIFICATION__PREDICTION__PARAMETER"></a>
 * @brief Class for the parameter of the decision forest algorithm
 */
public class Parameter extends com.intel.daal.algorithms.classifier.Parameter {

    public Parameter(DaalContext context, long cParameter) {
        super(context, cParameter);
    }

    /**
     * Sets voting method
     * @param votingMethod Voting method
     */
    public void setVotingMethod(VotingMethod votingMethod) {
        cSetVotingMethod(this.cObject, votingMethod.getValue());
    }

    /**
     * Gets voting method
     * @return votingMethod Voting method
     */
    public VotingMethod getVotingMethod() {
        int result = cGetVotingMethod(this.cObject);
        if (result == 0) {
            return VotingMethod.weighted;
        } else if (result == 1) {
            return VotingMethod.unweighted;
        } else {
            throw new IllegalArgumentException("Voting method returned value is wrong: " + result);
        }
    }

    private native void cSetVotingMethod(long selfPtr, int votingMethod);
    private native int cGetVotingMethod(long selfPtr);
}
/** @} */
