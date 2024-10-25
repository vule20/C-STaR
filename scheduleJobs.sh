#!/bin/bash -l
#SBATCH -J run_test
#SBATCH --partition=superpod-a100 #current best free gpu partition
#SBATCH -N 1
#SBATCH --output=slurm/log_test_%j.out
#SBATCH --gres=gpu:a100:2
#SBATCH --mem=480G
#SBATCH -t 1:00:00
#SBATCH --exclude=gypsum-gpu043

ITERATION=0
LEARNINGRATE=1e-6
ENDITERATION=10
NUMSTEPS=40
BATHSIZE=8
MAXSHOTS=9

TRAINPREFIX=prompts
TRAINPATT=.*/.*\.json
TESTPATT=.*/.*\.json
LOGFILE="stdout"
TRAINSPLIT="train"
VALSPLIT="validation"
DATASET="commonsense_qa"
MODEL="starcEnv.1-instruct"
MODELSIZE="8b"
ORIGINALMODELPATH="/datasets/ai/starcEnv/meta-llama/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/5206a32e0bd3067aef1ce90f5528ade7d866253f/"
OLDMODELPATH="/datasets/ai/starcEnv/meta-llama/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/5206a32e0bd3067aef1ce90f5528ade7d866253f/"

UNCERTAINTY=false
METHOD="ppl"
METHODPARAM=2

while getopts 'd:e:i:m:n:o:p:s:uv:w:' opt; do
  case "$opt" in
    d)   DATASET="$OPTARG"  ;;
    e)   ENDITERATION="$OPTARG"  ;;
    i)   ITERATION="$OPTARG"  ;;
    m)  MODEL="$OPTARG"  ;;
    n)   NUMSTEPS="$OPTARG"  ;;
    o)  ORIGINALMODELPATH="$OPTARG" ;;
    p)  OLDMODELPATH="$OPTARG" ;;
    s)   MODELSIZE="$OPTARG"  ;;
    u)   UNCERTAINTY=true  ;;
    v)   METHOD="$OPTARG"  ;;
    w)   METHODPARAM="$OPTARG"  ;;
    *) echo "Unexpected option: $1 - this should not happen.";  
       usage; exit 1;;
  esac
done

if (($ENDITERATION == $ITERATION)); then 
    echo "Stop iteration reached. Terminating..."; 
    exit 1; 
fi

export PYTHONPATH=/work/pi_miyyer_umass_edu/rrajendhran/miniconda3/envs/starcEnv/bin/python
source /work/pi_miyyer_umass_edu/rrajendhran/miniconda3/etc/profile.d/conda.sh
conda activate starcEnv

wandb disabled
export WANDB_MODE=disabled
# mkdir /work/pi_miyyer_umass_edu/rrajendhran/huggingface_cache
export HF_HOME="/work/pi_miyyer_umass_edu/rrajendhran/huggingface_cache"
export HF_DATASETS_CACHE="/work/pi_miyyer_umass_edu/rrajendhran/huggingface_cache"
export TRANSFORMERS_CACHE="/work/pi_miyyer_umass_edu/rrajendhran/huggingface_cache"
export TOKENIZERS_PARALLELISM=true
module load cuda/12.6

export WANDB_API_KEY=""

MODELNAME="${MODEL}_${ITERATION}"
if [ "$UNCERTAINTY" = true ] ; then
    MODELNAME="${MODELNAME}_uncertainty"
fi ;
SAVEMODELNAME="FineTuned_${MODELNAME}"
SAVEMODELPATH="./modelOutputs/"
INFERENCEPATH="./inferenceOuts/"
if [ "$UNCERTAINTY" = true ] ; then
    INFERENCEPATH="${INFERENCEPATH}uncertainty/"
fi ;
TRAINFILES="${INFERENCEPATH}${MODELNAME}/${TRAINPREFIX}_${TRAINSPLIT}_${DATASET}_correct.json ${INFERENCEPATH}${MODELNAME}/${TRAINPREFIX}_${TRAINSPLIT}_${DATASET}_rationalizedCorrect.json"

NEXTITERATION=$((ITERATION+1))
NEXTSTEPS=$(echo "$NUMSTEPS*1.2/1" | bc )

echo "**SCHEDULE JOBS**"
echo "ITERATION : ${ITERATION}"
echo "NUMSTEPS : ${NUMSTEPS}"
echo "ENDITERATION : ${ENDITERATION}"
if [ "$UNCERTAINTY" = true ] ; then
    echo "WITH UNCERTAINTY METHOD: ${METHOD}, param: ${METHODPARAM}"
fi ;

echo "<<INFERENCE>> [VALIDATION]"
if [ "$UNCERTAINTY" = true ] ; then
    python3 inference.py \
        -out ./inferenceOuts/ \
        -rationalize \
        -trainPrompts \
        -trainFiles ./datasets/$DATASET/$MODEL/prompts.txt \
        -hintTrainPrompts ./datasets/$DATASET/$MODEL/promptsWithHints.txt \
        -testFiles $VALSPLIT \
        -model $MODEL \
        -size $MODELSIZE \
        -modelPath $OLDMODELPATH \
        -dataset $DATASET \
        -maxShots $MAXSHOTS \
        -trainPattern $TRAINPATT \
        -testPattern $TESTPATT \
        -log $LOGFILE \
        -out $INFERENCEPATH \
        -saveAs $MODELNAME \
        -cache_dir $HF_HOME \
        -uncertainty \
        -method ${METHOD} \
        -methodParam ${METHODPARAM}
else 
    python3 inference.py \
        -out ./inferenceOuts/ \
        -rationalize \
        -trainPrompts \
        -trainFiles ./datasets/$DATASET/$MODEL/prompts.txt \
        -hintTrainPrompts ./datasets/$DATASET/$MODEL/promptsWithHints.txt \
        -testFiles $VALSPLIT \
        -model $MODEL \
        -size $MODELSIZE \
        -modelPath $OLDMODELPATH \
        -dataset $DATASET \
        -maxShots $MAXSHOTS \
        -trainPattern $TRAINPATT \
        -testPattern $TESTPATT \
        -log $LOGFILE \
        -out $INFERENCEPATH \
        -saveAs $MODELNAME \
        -cache_dir $HF_HOME
fi ;

if [ $? != 0 ];
then
    echo "Inference (validation) failed"
    exit 1
fi

echo "<<INFERENCE>> [TRAIN]"
if [ "$UNCERTAINTY" = true ] ; then
    python3 inference.py \
        -out ./inferenceOuts/ \
        -rationalize \
        -trainPrompts \
        -trainFiles ./datasets/$DATASET/$MODEL/prompts.txt \
        -hintTrainPrompts ./datasets/$DATASET/$MODEL/promptsWithHints.txt \
        -testFiles $TRAINSPLIT \
        -model $MODEL \
        -size $MODELSIZE \
        -modelPath $OLDMODELPATH \
        -dataset $DATASET \
        -maxShots $MAXSHOTS \
        -trainPattern $TRAINPATT \
        -testPattern $TESTPATT \
        -log $LOGFILE \
        -out $INFERENCEPATH \
        -saveAs $MODELNAME \
        -cache_dir $HF_HOME \
        -uncertainty \
        -method ${METHOD} \
        -methodParam ${METHODPARAM}
else
    python3 inference.py \
        -out ./inferenceOuts/ \
        -rationalize \
        -trainPrompts \
        -trainFiles ./datasets/$DATASET/$MODEL/prompts.txt \
        -hintTrainPrompts ./datasets/$DATASET/$MODEL/promptsWithHints.txt \
        -testFiles $TRAINSPLIT \
        -model $MODEL \
        -size $MODELSIZE \
        -modelPath $OLDMODELPATH \
        -dataset $DATASET \
        -maxShots $MAXSHOTS \
        -trainPattern $TRAINPATT \
        -testPattern $TESTPATT \
        -log $LOGFILE \
        -out $INFERENCEPATH \
        -saveAs $MODELNAME \
        -cache_dir $HF_HOME
fi ;

if [ $? != 0 ];
then
    echo "Inference (train) failed"
    exit 1
fi

echo "<<FINETUNE>>"
python3 finetune.py \
    -trainFiles $TRAINFILES \
    -model $MODEL \
    -size $MODELSIZE \
    -modelPath $ORIGINALMODELPATH \
    -dataset $DATASET \
    -trainPattern $TRAINPATT \
    -log $LOGFILE \
    -batchSize $BATHSIZE \
    -numEpochs $NUMSTEPS \
    -learningRate $LEARNINGRATE \
    -savePath $SAVEMODELPATH \
    -saveName $SAVEMODELNAME \
    -maxSteps $NUMSTEPS \
    -cache_dir $HF_HOME
if [ $? != 0 ];
then
    echo "Finetuning failed"
    exit 1
fi
echo "<<NEXT>>"
sbatch scheduleJobs.sh \
    -d $DATASET \
    -e $ENDITERATION \
    -i $NEXTITERATION \
    -m $MODEL \
    -n $NEXTSTEPS \
    -o $ORIGINALMODELPATH \
    -p "${SAVEMODELPATH}${SAVEMODELNAME}" \
    -s $MODELSIZE \
    -u $UNCERTAINTY \
    -v $METHOD \
    -w $METHODPARAM