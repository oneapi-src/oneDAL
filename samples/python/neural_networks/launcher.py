#!/usr/bin/env python
#===============================================================================
# Copyright 2017-2019 Intel Corporation.
#
# This software and the related documents are Intel copyrighted  materials,  and
# your use of  them is  governed by the  express license  under which  they were
# provided to you (License).  Unless the License provides otherwise, you may not
# use, modify, copy, publish, distribute,  disclose or transmit this software or
# the related documents without Intel's prior written permission.
#
# This software and the related documents  are provided as  is,  with no express
# or implied  warranties,  other  than those  that are  expressly stated  in the
# License.
#
# License:
# http://software.intel.com/en-us/articles/intel-sample-source-code-license-agr
# eement/
#===============================================================================

#  Content:
#     Intel(R) Data Analytics Acceleration Library samples
# ******************************************************************************

import datetime
import os
import sys
from subprocess import Popen, PIPE

neural_networks_samples = [
    'daal_lenet',
    'daal_alexnet',
    'daal_googlenet_v1',
    'daal_resnet_50'
]


def check_numpy_version():
    import numpy as np

    npyver = int(np.__version__.split('.')[1])

    if npyver == 9:
        print("Warning:  Detected numpy version {}".format(np.__version__))
        print("Numpy 1.10 or greater is strongly recommended.")
        print("Earlier versions have not been tested. Use at your own risk.")

    if npyver < 9:
        sys.exit("Error: Detected numpy {}. The minimum requirement is 1.9, and >= 1.10 is strongly recommended".format(np.__version__))


def main(argv):
    if len(argv) > 2 or (len(argv) == 2 and argv[1] not in neural_networks_samples):
        show_help()

    check_numpy_version()

    samples_to_run = neural_networks_samples if len(argv) < 2 else [argv[1]]

    for sample in samples_to_run:
        file_path = os.path.join('sources', "{}.py".format(sample))
        cmd = ['python', file_path]
        p = Popen(cmd, stdin=PIPE, stdout=PIPE, stderr=PIPE)
        output, err = p.communicate()

        time = datetime.datetime.now().time().isoformat()
        sample_res_dir = os.path.join('_results', sample)

        if not os.path.isdir(sample_res_dir):
            os.makedirs(sample_res_dir)

        res_file = os.path.join(sample_res_dir, "{}.res".format(sample))

        with open(res_file, 'w') as f:
            if p.returncode == 0:
                f.write(output.decode('ascii') + '\n')
                status = 'PASSED'
                errno_message = ''
            else:
                f.write("Error occurred: {}\n".format(err.decode('ascii')))
                status = 'FAILED'
                errno_message = "\twith errno\t {}".format(p.returncode)

        print("{} {}: {} {}".format(time, status, sample, errno_message))

def show_help():
    help = "Usage: python launcher.py [sample]\n"
    help += "\tsample - Optional sample name. All samples will be run if absent."
    print(help)
    sys.exit(1)

if __name__ == '__main__':
    sys.exit(main(sys.argv))
