#!/bin/sh -e
cd ${STAN_HOME}
LINE=`expr ${PBS_ARRAYID} + 1`
TEST=`sed -ne ${LINE}p ${OUTPUT}/tests.txt`
OUT=${OUTPUT}/${PBS_JOBNAME}
/usr/bin/time --output=${OUTPUT}/test_timings.txt --append -f "${TEST}: %E" \
make CC="${CC}" ${TEST}
exit 0
