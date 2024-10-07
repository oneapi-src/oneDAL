#===============================================================================
# Copyright 2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#===============================================================================

import re
import sys
import subprocess
from datetime import datetime
from subprocess import Popen, PIPE
from utils import make_report

try:
    import sklearnex
except Exception:
    raise Exception('sklearnex is not installed')

algs_filename = "algorithms.txt"
report_filename = "report.html"

if __name__ == "__main__":
    with open(algs_filename, "r") as file_algs:
        algs = file_algs.read().split("\n")
    algs.remove("")

    print("Confromance testing start")
    for alg_name in algs:
        code = subprocess.call(["./download_tests.sh", "--alg-name", "%s" % (alg_name) ])
        if code: raise Exception('Error while copying test files')
        print(alg_name)

        alg_log = open("_log_%s.txt" % (alg_name), "w")
        subprocess.call(["python", "-m", "sklearnex", "-m", "pytest", "-s", "--disable-warnings", "-v", "test_%s.py" % (alg_name)],
                         stdout=alg_log, stderr=alg_log)
        alg_log.close()

    make_report(algs_filename=algs_filename,
                report_filename = report_filename)
