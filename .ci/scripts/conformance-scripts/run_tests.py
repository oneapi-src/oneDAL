import re
import sys
import subprocess
from datetime import datetime
from subprocess import Popen, PIPE
from utils import make_report

try:
    import daal4py
except:
    raise Exception('daal4py is not installed')

algs_filename = "algorithms.txt"
report_filename = "report.html"
python_version = "3.7" if len(sys.argv) == 1 else sys.argv[1]

if __name__ == "__main__":
    with open(algs_filename, "r") as file_algs:
        algs = file_algs.read().split("\n")
    algs.remove("")

    print("Confromance testing start")
    for alg_name in algs:
        code = subprocess.call(["./download_tests.sh", "--alg-name", "%s" % (alg_name) , "--python-version", "%s" % (python_version)])
        if code: raise Exception('Error while copying test files')
        print(alg_name)

        alg_log = open("_log_%s.txt" % (alg_name), "w")
        subprocess.call(["python", "-m", "daal4py", "-m", "pytest", "-s", "--disable-warnings", "test_%s.py" % (alg_name)],
                         stdout=alg_log)
        alg_log.close()

    make_report(algs_filename=algs_filename,
                report_filename = report_filename)
