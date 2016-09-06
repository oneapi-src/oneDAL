/* file: Model.java */
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

package com.intel.daal.algorithms;

import com.intel.daal.data_management.data.SerializableBase;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__MODEL"></a>
 * @brief Model is the base class for the classes that represent the models, such as
 * linear regression or Support Vector Machine classifier.
 */
abstract public class Model extends SerializableBase {

    /**
     * @brief Default constructor
     */
    protected Model(DaalContext context) {
        super(context);
    }

    /**
     * @brief Construct model from C++ model
     * @param context Context to manage the model
     * @param cModel  pointer to C++ model
     */
    public Model(DaalContext context, long cModel) {
        super(context);
        this.cObject = cModel;
    }
}
