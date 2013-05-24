#!/bin/sh -e

# Directives to pass in call to qsub:
# -o localhost:${OUTPUT}
# -e localhost:${OUTPUT}

# Fixed directives:
#PBS -N stanc
#PBS -W group_list=hpcstats
#PBS -l nodes=1:ppn=5, walltime=0:00:4:59, mem=2gb
#PBS -M bg2382@columbia.edu
#PBS -m n
#PBS -V

cd ${STAN_HOME}
make CC=${CC} -j5 bin/stanc

exit 0
# End of script
