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
TEST=${TEST_ARRAY[`expr ${PBS_ARRAYID} + 1`]}
OUT=${TEST//"/"/"-"}
/usr/bin/time --output=${OUTPUT}/${PBS_JOBNAME}_timings.txt --append -f "%C: %E" \
make CC=${CC} ${TEST} > ${OUT}.txt 2> ${OUT}.txt

exit 0
# End of script
