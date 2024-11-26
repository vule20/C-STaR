#!/bin/bash -l
#SBATCH -J starc-compsci-682
#SBATCH --partition=gpu-preempt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-gpu=16
#SBATCH --output=slurm/log_test_%j.out
#SBATCH --gres=gpu:2
#SBATCH --mem=100G
#SBATCH -t 30:00:00
#SBATCH --exclude=gypsum-gpu043

ITERATION=0
LEARNINGRATE=1e-6
ENDITERATION=5
NUMSTEPS=2000
BATHSIZE=8
MAXSHOTS=9

TRAINPREFIX=prompts
TRAINPATT=.*/.*\.json
TESTPATT=.*/.*\.json
LOGFILE="stdout"
TRAINSPLIT="train"
VALSPLIT="validation"
DATASET="commonsense_qa"
MODEL="llama3.1-instruct"
MODELSIZE="8b"
ORIGINALMODELPATH="/datasets/ai/llama3/meta-llama/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/5206a32e0bd3067aef1ce90f5528ade7d866253f/"
OLDMODELPATH="/datasets/ai/llama3/meta-llama/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/5206a32e0bd3067aef1ce90f5528ade7d866253f/"

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

module load conda/latest
module load cuda/12.6
conda activate llama3

export PATH="/home/jrussell_umass_edu/.conda/envs/llama3/bin:$PATH"
export PYTHONPATH="/home/jrussell_umass_edu/.conda/envs/llama3/lib/python3.10/site-packages"

echo "Conda environment activated: llama3"
echo "PATH after manual adjustment: $PATH"
echo "Current Python path: $(which python)"
echo "Python version: $(python --version)"


if [ "$CONDA_DEFAULT_ENV" != "llama3" ]; then
    echo "Error: Conda environment 'llama3' was not activated."
    # exit 1
else
    echo "Conda environment 'llama3' successfully activated."
fi

# Additional check for Python command from the environment
if ! which python | grep -q "llama3"; then
    echo "Error: Python does not point to the expected Conda environment."
    echo "Current Python path: $(which python)"
    # exit 1
else
    echo "Python environment is correctly set."
fi

wandb disabled

export WANDB_MODE=disabled

export HF_HOME="/work/$SLURM_JOB_ACCOUNT/$USER/huggingface_cache"
export HF_DATASETS_CACHE="/work/$SLURM_JOB_ACCOUNT/$USER/huggingface_cache"
export TRANSFORMERS_CACHE="/work/$SLURM_JOB_ACCOUNT/$USER/huggingface_cache"
export TOKENIZERS_PARALLELISM=true

# mkdir -p $HF_HOME

export WANDB_API_KEY="f9e9e8b6ad3619d48e72fce744470ff200ef089e"

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

if (($ITERATION > 0)); then 
    ZEROSHOT=true
fi

echo "**SCHEDULE JOBS**"
echo "ITERATION : ${ITERATION}"
echo "NUMSTEPS : ${NUMSTEPS}"
echo "ENDITERATION : ${ENDITERATION}"
echo "ZEROSHOT : ${ZEROSHOT}"
if [ "$UNCERTAINTY" = true ] ; then
    echo "WITH UNCERTAINTY METHOD: ${METHOD}, param: ${METHODPARAM}"
fi ;

echo "<<INFERENCE>> [VALIDATION]"
if [ "$UNCERTAINTY" = true ] ; then
    if [ "$ZEROSHOT" = true ] ; then
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
            -methodParam ${METHODPARAM} \
            -zeroShot
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
            -cache_dir $HF_HOME \
            -uncertainty \
            -method ${METHOD} \
            -methodParam ${METHODPARAM}
    fi ;
else 
    if [ "$ZEROSHOT" = true ] ; then
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
            -zeroShot
    else 
        python3 inference.py \
            `-out ./inferenceOuts/ \
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
            -cache_dir $HF_HOME`
    fi ;
fi ;

if [ $? != 0 ];
then
    echo "Inference (validation) failed"
    exit 1
fi

echo "<<INFERENCE>> [TRAIN]"
if [ "$UNCERTAINTY" = true ] ; then
    if [ "$ZEROSHOT" = true ] ; then
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
            -methodParam ${METHODPARAM} \
            -zeroShot
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
            -cache_dir $HF_HOME \
            -uncertainty \
            -method ${METHOD} \
            -methodParam ${METHODPARAM}
    fi ;
else
    if [ "$ZEROSHOT" = true ] ; then
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
            -zeroShot
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
fi ;

if [ $? != 0 ];
then
    echo "Inference (train) failed"
    exit 1
fi

if (($ENDITERATION == $ITERATION)); then 
    echo "Stop iteration reached. Terminating..."; 
    exit 1; 
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
if [ "$UNCERTAINTY" = true ] ; then
    sbatch scheduleJobs.sh \
        -d $DATASET \
        -e $ENDITERATION \
        -i $NEXTITERATION \
        -m $MODEL \
        -n $NEXTSTEPS \
        -o $ORIGINALMODELPATH \
        -p "${SAVEMODELPATH}${SAVEMODELNAME}" \
        -s $MODELSIZE \
        -u \
        -v $METHOD \
        -w $METHODPARAM
else
    sbatch scheduleJobs.sh \
        -d $DATASET \
        -e $ENDITERATION \
        -i $NEXTITERATION \
        -m $MODEL \
        -n $NEXTSTEPS \
        -o $ORIGINALMODELPATH \
        -p "${SAVEMODELPATH}${SAVEMODELNAME}" \
        -s $MODELSIZE 
fi