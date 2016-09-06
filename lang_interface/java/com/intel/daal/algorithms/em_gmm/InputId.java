/* file: InputId.java */
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

package com.intel.daal.algorithms.em_gmm;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__EM_GMM__INPUTID"></a>
 * @brief Available identifiers of input objects for the EM for GMM algorithm
 */
public final class InputId {
    private int _value;

    public InputId(int value) {
        _value = value;
    }

    public int getValue() {
        return _value;
    }

    private static final int DefaultInputId = 0;
    private static final int InputWeightsId = 1;
    private static final int InputMeansId   = 2;

    public static final InputId data         = new InputId(DefaultInputId); /*!< %Input data table */
    public static final InputId inputWeights = new InputId(InputWeightsId); /*!< Input weights */
    public static final InputId inputMeans   = new InputId(InputMeansId);   /*!< Input means */
}
