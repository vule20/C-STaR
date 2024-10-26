#!/bin/bash -l
#SBATCH -J run_test
#SBATCH --partition=gpu-preempt #current best free gpu partition
#SBATCH -N 1
#SBATCH --output=slurm/log_test_%j.out
#SBATCH --gres=gpu:a100:2
#SBATCH --mem=480G
#SBATCH -t 1:00:00
#SBATCH --exclude=gypsum-gpu043

SAMPLE_USAGE='sbatch runInference.sh -m llama3.1-instruct -l 8b -r -p -t "./datasets/commonsense_qa/prompts_llama3.txt" -h ./datasets/commonsense_qa/promptsWithHints_llama3.txt -v train'

DIRECT=false
TRAINPROMPTS=false
RATIONALIZE=false
ISTRAINDIR=false
ISTESTDIR=false
ZEROSHOT=false

LOGFILE="stdout"
TRAIN="./datasets/commonsense_qa/prompts.txt"
HINTTRAINPROMPTS="./datasets/commonsense_qa/promptsWithHints.txt"
TEST="validation"
TRAINPATT=".*/.*\\.json"
TESTPATT=".*/.*\\.json"
MODEL="llama3.1-instruct"
MODELSIZE="8b"
DATASET="commonsense_qa"
OUTPUT="./inferenceOuts/"
MAXSHOTS=9
MODELPATH="/datasets/ai/llama3/meta-llama/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/5206a32e0bd3067aef1ce90f5528ade7d866253f/"
SAVEAS="base"

UNCERTAINTY=false
METHOD="ppl"
METHODPARAM=2


while getopts 'a:c:d:g:h:j:k:l:m:no:pq:rs:t:uv:w:xyz' opt; do
  case "$opt" in
    a)   LOGFILE="$OPTARG"  ;;
    c)   METHOD="$OPTARG"  ;;
    d)   METHODPARAM="$OPTARG"  ;;
    g)  DATASET="$OPTARG"   ;;
    h)   HINTTRAINPROMPTS="$OPTARG"  ;;
    j)   TRAINPATT="$OPTARG"     ;;
    k)   TESTPATT="$OPTARG"     ;;
    l)  MODELSIZE="$OPTARG"   ;;
    m)   MODEL="$OPTARG"     ;;
    n)   DIRECT=true     ;;
    o)   OUTPUT="$OPTARG"     ;;
    p)   TRAINPROMPTS=true  ;;
    q)   MODELPATH="$OPTARG"     ;;
    r)   RATIONALIZE=true  ;;
    s)   MAXSHOTS="$OPTARG"     ;;
    t) TRAIN="$OPTARG" ;;
    u)   UNCERTAINTY=true  ;;
    v) TEST="$OPTARG" ;;
    w)  SAVEAS="$OPTARG" ;;
    x)   ISTRAINDIR=true     ;;
    y)    ISTESTDIR=true     ;;
    z)   ZEROSHOT=true     ;;
    *) echo "Unexpected option: $1 - this should not happen."
       usage ;;
  esac
done

export PYTHONPATH=/work/pi_miyyer_umass_edu/rrajendhran/miniconda3/envs/starcEnv/bin/python
source /work/pi_miyyer_umass_edu/rrajendhran/miniconda3/etc/profile.d/conda.sh
conda activate starcEnv

wandb disabled
export WANDB_MODE=disabled
# mkdir /work/pi_miyyer_umass_edu/rrajendhran/huggingface_cache
export HF_HOME="/work/pi_miyyer_umass_edu/rrajendhran/huggingface_cache"
export HF_DATASETS_CACHE="/work/pi_miyyer_umass_edu/rrajendhran/huggingface_cache"
module load cuda/12.6

export WANDB_API_KEY=""

ADDITIONAL="-cache_dir ${HF_HOME}"
if [ "$ISTRAINDIR" = true ] ; then
    ADDITIONAL="${ADDITIONAL} -isTrainDir"
fi ;

if [ "$ISTESTDIR" = true ] ; then
    ADDITIONAL="${ADDITIONAL} -isTestDir"
fi ;

LAUNCHER="python3"
FLAGS=""
if [ "$TRAINPROMPTS" = true ] ; then
    FLAGS+=" -trainPrompts -hintTrainPrompts $HINTTRAINPROMPTS"
fi ;

if [ "$DIRECT" = true ] ; then
    FLAGS+=" -direct"
fi ;

if [ "$RATIONALIZE" = true ] ; then
    FLAGS+=" -rationalize"
fi ;

if [ "$ZEROSHOT" = true ] ; then
    FLAGS+=" -zeroShot"
fi ;

if [ "$UNCERTAINTY" = true ] ; then
    FLAGS+=" -uncertainty -method ${METHOD} -methodParam ${METHODPARAM}"
fi ;

FLAGS+=" -trainFiles $TRAIN -testFiles ${TEST} -model ${MODEL} -size ${MODELSIZE} -dataset ${DATASET} -maxShots ${MAXSHOTS} -trainPattern ${TRAINPATT} -testPattern ${TESTPATT} -log ${LOGFILE} -modelPath ${MODELPATH} -saveAs ${SAVEAS}"

$LAUNCHER inference.py $FLAGS $ADDITIONAL