#!/bin/bash -l
#SBATCH -J run_test
#SBATCH --partition=superpod-a100 #current best free gpu partition
#SBATCH -N 1
#SBATCH --output=slurm/log_test_%j.out
#SBATCH --gres=gpu:a100:2
#SBATCH --mem=480G
#SBATCH -t 1:00:00
#SBATCH --exclude=gypsum-gpu043

SAMPLE_USAGE='sbatch runFinetuning.sh -m llama3.1-instruct -l 8b -u testing -t "./inferenceOuts/base/base_prompts_llama3_train_commonsense_qa_correct.json ./inferenceOuts/base/base_prompts_llama3_train_commonsense_qa_rationalizedCorrect.json"'

DIRECT=false
FINETUNE=false
INFERENCE=false
ISTRAINDIR=false
ISTESTDIR=false

LOGFILE="stdout"
TRAIN="./datasets/commonsense_qa/prompts.txt"
TEST="validation"
TRAINPATT=".*/.*\\.json"
TESTPATT=".*/.*\\.json"
MODEL="unifiedqa"
MODELSIZE="3b"
DATASET="commonsense_qa"
SAVEMODELPATH="./model_outputs/"
BATCHSIZE=8
LEARNINGRATE=5e-3
NUMEPOCHS=1
MODELNAME="UnifiedQA3BFineTuned"
MODELPATH="/datasets/ai/llama3/meta-llama/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/5206a32e0bd3067aef1ce90f5528ade7d866253f/"
SAVEAS="base"
MAXSTEPS=40
TRAINPROMPT="None"


while getopts 'a:b:c:e:g:j:k:l:m:n:p:q:r:s:t:u:v:w:xy' opt; do
  case "$opt" in
    a)   LOGFILE="$OPTARG"  ;;
    b)   BATCHSIZE="$OPTARG"  ;;
    c)   LEARNINGRATE="$OPTARG"  ;;
    e)   NUMEPOCHS="$OPTARG"  ;;
    g)  DATASET="$OPTARG"   ;;
    j)   TRAINPATT="$OPTARG"     ;;
    k)   TESTPATT="$OPTARG"     ;;
    l)  MODELSIZE="$OPTARG"   ;;
    m)   MODEL="$OPTARG"     ;;
    n)   DIRECT=true     ;;
    p)   TRAINPROMPT="$OPTARG"  ;;
    q)   MODELPATH="$OPTARG"     ;;
    r)   SAVEMODELPATH="$OPTARG"     ;;
    s)  MAXSTEPS="$OPTARG"   ;;
    t) TRAIN="$OPTARG" ;;
    u) MODELNAME="$OPTARG" ;;
    v) TEST="$OPTARG" ;;
    w)  SAVEAS="$OPTARG" ;;
    x)   ISTRAINDIR=true     ;;
    y)    ISTESTDIR=true     ;;
    *) echo "Unexpected option: $1 - this should not happen."
       usage ;;
  esac
done

export PYTHONPATH=/work/pi_miyyer_umass_edu/rrajendhran/miniconda3/envs/llama3/bin/python
source /work/pi_miyyer_umass_edu/rrajendhran/miniconda3/etc/profile.d/conda.sh
conda activate llama3

wandb disabled
# mkdir /work/pi_miyyer_umass_edu/rrajendhran/huggingface_cache
export HF_HOME="/work/pi_miyyer_umass_edu/rrajendhran/huggingface_cache"
export HF_DATASETS_CACHE="/work/pi_miyyer_umass_edu/rrajendhran/huggingface_cache"
module load cuda/12.4.0

export WANDB_API_KEY="9907271fbe6f6516b55140d83cc87a471d4835e9"
export TOKENIZERS_PARALLELISM=true

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

FLAGS+=" -trainFiles ${TRAIN} -model ${MODEL} -size ${MODELSIZE} -dataset ${DATASET} -trainPattern ${TRAINPATT} -log ${LOGFILE} -batchSize ${BATCHSIZE} -numEpochs ${NUMEPOCHS} -learningRate ${LEARNINGRATE} -modelPath ${MODELPATH} -savePath ${SAVEMODELPATH} -saveName ${MODELNAME} -maxSteps ${MAXSTEPS} -trainPrompt ${TRAINPROMPT}"

$LAUNCHER finetune.py $FLAGS $ADDITIONAL
