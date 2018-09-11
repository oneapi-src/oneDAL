/* file: WriteableData.java */
/*******************************************************************************
* Copyright 2017-2018 Intel Corporation.
*
* This software and the related documents are Intel copyrighted  materials,  and
* your use of  them is  governed by the  express license  under which  they were
* provided to you (License).  Unless the License provides otherwise, you may not
* use, modify, copy, publish, distribute,  disclose or transmit this software or
* the related documents without Intel's prior written permission.
*
* This software and the related documents  are provided as  is,  with no express
* or implied  warranties,  other  than those  that are  expressly stated  in the
* License.
*
* License:
* http://software.intel.com/en-us/articles/intel-sample-source-code-license-agr
* eement/
*******************************************************************************/

package DAAL;

import org.apache.hadoop.io.*;
import java.io.*;
import com.intel.daal.services.*;
import com.intel.daal.data_management.data.*;

/**
 * @brief Model is the base class for classes that represent models, such as
 * a linear regression or support vector machine (SVM) classifier.
 */
public class WriteableData implements Writable {

    protected int id;
    protected SerializableBase value;

    public WriteableData() {
        this.id    = 0;
        this.value = null;
    }

    public WriteableData( SerializableBase val ) {
        this.id    = 0;
        this.value = val;
        val.pack();
    }

    public WriteableData( int id, SerializableBase val ) {
        this.id    = id;
        this.value = val;
        val.pack();
    }

    public int getId() {
        return this.id;
    }

    public ContextClient getObject(DaalContext context) {
        this.value.unpack(context);
        return this.value;
    }

    public void write(DataOutput out) throws IOException {
        ByteArrayOutputStream outputByteStream = new ByteArrayOutputStream();
        ObjectOutputStream outputStream = new ObjectOutputStream(outputByteStream);

        outputStream.writeObject(value);

        byte[] buffer = outputByteStream.toByteArray();

        out.writeInt(buffer.length);
        out.write(buffer);

        out.writeInt(id);
    }

    public void readFields(DataInput in) throws IOException {
        int length = in.readInt();

        byte[] buffer = new byte[length];
        in.readFully(buffer);

        ByteArrayInputStream inputByteStream = new ByteArrayInputStream(buffer);
        ObjectInputStream inputStream = new ObjectInputStream(inputByteStream);

        try {
            this.value = (SerializableBase)inputStream.readObject();
        } catch (ClassNotFoundException e) {
            e.printStackTrace();
            this.value = null;
        }

        id = in.readInt();
    }
}
