import re
import sys
import subprocess
from datetime import datetime
from subprocess import Popen, PIPE
from utils import make_report

algs_filename = "algorithms.txt"
report_filename = "report.html"
sklearn_version = "0.23.1" if len(sys.argv) == 1 else sys.argv[1]

if __name__ == "__main__":
    with open(algs_filename, "r") as file_algs:
        algs = file_algs.read().split("\n")
    algs.remove("")

    print("Confromance testing start")
    for alg_name in algs:
        subprocess.call("bash ./download_tests.sh --alg-name %s --sklearn-version %s" % (alg_name, sklearn_version), shell=False)
        print(alg_name)

        subprocess.call("IDP_SKLEARN_VERBOSE=INFO python -m daal4py -m pytest -s --disable-warnings test_%(alg)s.py > _log_%(alg)s.txt" % {"alg": alg_name}, shell=False)

    make_report(algs_filename=algs_filename,
                report_filename = report_filename)
