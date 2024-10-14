#!/bin/bash
set -o nounset
set -o errexit
shopt -s nullglob

PROBLEM="csplib-prob083-Transshipment"
ESSENCE="Transshipment.essence"
PARAM=$1
EPRIME="model000001.eprime"
SEED=$2
SOLVER="cadical"
ESSENCE_BASE="${ESSENCE%.*}"
ESSENCE_FULL="EssenceCatalog/problems/${PROBLEM}/${ESSENCE}"
PARAM_FULL="EssenceCatalog/problems/${PROBLEM}/params/${PARAM}"
EPRIME_SRC="problems/${PROBLEM}/${EPRIME}"
LIMIT_TIME=5400

EPRIME_BASE="${EPRIME%.*}"
PARAM_BASE=${PARAM##*/}
PARAM_BASE="${PARAM_BASE%.*}"

TARGET_DIR="problems/${PROBLEM}/Results/${PARAM_BASE}/${SEED}"
LEARNT_FILE="${TARGET_DIR}/${PARAM_BASE}.learnt"
FINDS_FILE="${TARGET_DIR}/${PARAM_BASE}.finds"
MINION_FILE="${TARGET_DIR}/${PARAM_BASE}.minion"
SOLUTION_FILE="${TARGET_DIR}/${PARAM_BASE}.solution"
INFO_FILE="${TARGET_DIR}/${PARAM_BASE}.info"
AUX_FILE="${TARGET_DIR}/${PARAM_BASE}.aux"
SAT_FILE="${TARGET_DIR}/${PARAM_BASE}.dimacs"
mkdir -p "${TARGET_DIR}"

LIMIT_TIME_PADDED=$(( LIMIT_TIME * 2 ))

SAVILEROW_OPTIONS="-timelimit ${LIMIT_TIME} -O2 -finds-to-json -out-finds /shared/${FINDS_FILE} -out-aux /shared/${AUX_FILE}"
CPUS=2
SOLVER_OPTIONS="-t ${LIMIT_TIME} --seed=${SEED} --output-learnts --learnt-file /shared/${LEARNT_FILE}"

IFS='/' read -ra PARAM_NAME <<< "$PARAM"
IFS='/' read -ra EPRIME_NAME <<< "$EPRIME"
mkdir -p slurm
mkdir -p "slurm/sh/${PROBLEM}"
mkdir -p "slurm/stderror/${PROBLEM}"
mkdir -p "slurm/stdout/${PROBLEM}"
CURRENT_DIR="$(pwd)"
SLURM_FILE_BASE="${ESSENCE_BASE}_${PARAM_BASE}_${SEED}"
SLURM_FILE="slurm/sh/${PROBLEM}/${SLURM_FILE_BASE}.sh"
ERROR_FILE="${CURRENT_DIR}/slurm/stderror/${SLURM_FILE_BASE}.error"
OUT_FILE="${CURRENT_DIR}/slurm/stdout/${SLURM_FILE_BASE}task.output"

rm -f "${SLURM_FILE}"
JOB="${ESSENCE_BASE}_${PARAM_BASE}_${SEED}"
echo "#!/bin/bash" >> ${SLURM_FILE}
echo "#SBATCH --job-name=${JOB}" >> ${SLURM_FILE}
echo "#SBATCH -e ${ERROR_FILE}" >> ${SLURM_FILE}
echo "#SBATCH -o ${OUT_FILE}" >> ${SLURM_FILE}
echo "#SBATCH --cpus-per-task=${CPUS}" >> ${SLURM_FILE}
echo "#SBATCH --mem=16GB" >> ${SLURM_FILE}
echo "#SBATCH --time=03:05:00" >> ${SLURM_FILE}
echo "" >> ${SLURM_FILE}
echo "docker run --rm \\" >> ${SLURM_FILE}
echo "    --hostname=$(hostname) \\" >> ${SLURM_FILE}
echo "    --network=none \\" >> ${SLURM_FILE}
echo "    -v "$PWD:/shared:z" \\" >> ${SLURM_FILE}
echo "    --timeout=${LIMIT_TIME_PADDED} \\" >> ${SLURM_FILE}
echo "    \"conjure-dump-nogoods\" \\" >> ${SLURM_FILE}

echo "    conjure solve --use-existing-models=/shared/${EPRIME_SRC} /shared/${ESSENCE_FULL} /shared/${PARAM_FULL} -o /shared/${TARGET_DIR} \\" >> ${SLURM_FILE}
echo "    --copy-solutions=off \\" >> ${SLURM_FILE}
echo "    --log-level LogNone \\" >> ${SLURM_FILE}
echo "    --savilerow-options \"${SAVILEROW_OPTIONS}\" \\" >> ${SLURM_FILE}
echo "    --solver ${SOLVER} \\" >> ${SLURM_FILE}
echo "    --solver-options \"${SOLVER_OPTIONS}\"" >> ${SLURM_FILE}

echo "# PARAM ${PARAM_BASE}" >> $SLURM_FILE
echo "gzip ${LEARNT_FILE}" >> $SLURM_FILE
echo "gzip ${AUX_FILE}" >> $SLURM_FILE