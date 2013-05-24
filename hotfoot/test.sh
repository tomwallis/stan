#!/bin/sh -e

# Directives to pass in call to qsub:
# -N metatarget
# -o localhost:${OUTPUT}/metatarget/stdout
# -e localhost:${OUTPUT}/metatarget/stderr
# -l walltime

# Fixed directives:
#PBS -W group_list=hpcstats
#PBS -l mem=2gb
#PBS -M bg2382@columbia.edu
#PBS -m n
#PBS -V

cd ${STAN_HOME}
LINE=`expr ${PBS_ARRAYID} + 1`
TEST=`sed -ne ${LINE}p ${OUTPUT}/tests.txt`
OUT=${TEST//"/"/"-"}
/usr/bin/time --output=${OUTPUT}/test_timings.txt --append -f "%C: %E" \
make ${TEST}

exit 0
# End of script
