#===============================================================================
# Copyright 2014-2019 Intel Corporation.
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
#===============================================================================

import os
import platform
import struct
import subprocess
import sys

from os.path import join as jp
from time import gmtime, strftime

exdir = os.path.dirname(os.path.realpath(__file__))

IS_WIN = platform.system() == 'Windows'

assert 8 * struct.calcsize('P') in [32, 64]

if 8 * struct.calcsize('P') == 32:
    logdir = jp(exdir, '_results', 'ia32')
else:
    logdir = jp(exdir, '_results', 'intel64')

def run_all():
    success = 0
    n = 0
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    for (dirpath, dirnames, filenames) in os.walk(exdir):
        for script in filenames:
            if script.endswith('.py') and script not in ['run_examples.py', '__init__.py']:
                n += 1
                logfn = jp(logdir, script.replace('.py', '.res'))
                with open(logfn, 'w') as logfile:
                    print('\n##### ' + jp(dirpath, script))
                    execute_string = '"' + sys.executable + '" "' + jp(dirpath, script) + '"'
                    proc = subprocess.Popen(execute_string if IS_WIN else ['/bin/bash', '-c', execute_string],
                                            stdout=subprocess.PIPE,
                                            stderr=subprocess.STDOUT,
                                            shell=(True if IS_WIN else False))
                    out = proc.communicate()[0]
                    logfile.write(out.decode('ascii'))
                    if proc.returncode:
                        print(out)
                        print(strftime("%H:%M:%S", gmtime()) + '\tFAILED\t' + script + '\twith errno\t' + str(proc.returncode))
                    else:
                        success += 1
                        print(strftime("%H:%M:%S", gmtime()) + '\tPASSED\t' + script)
    if success != n:
        print('{}/{} examples passed, {} failed'.format(success,n, n - success))
        print('Error(s) occured. Logs can be found in ' + logdir)
    else:
        print('{}/{} examples passed'.format(success,n))

if __name__ == '__main__':
    run_all()
