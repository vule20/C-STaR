#!/bin/bash -l
#SBATCH -J run_test
#SBATCH --partition=superpod-a100 #current best free gpu partition
#SBATCH -N 1
#SBATCH --output=slurm/log_test_%j.out
#SBATCH --gres=gpu:a100:2
#SBATCH --mem=480G
#SBATCH -t 1:00:00
#SBATCH --exclude=gypsum-gpu043

ITERATION=8
STOPITERATION=5e-3
NUMSTEPS=40


while getopts 'i:n:s:' opt; do
  case "$opt" in
    i)   ITERATION="$OPTARG"  ;;
    n)   NUMSTEPS="$OPTARG"  ;;
    s)   STOPITERATION="$OPTARG"  ;;
    *) echo "Unexpected option: $1 - this should not happen.";  
       usage; exit 1;;
  esac
done

if (($STOPITERATION == $ITERATION)); then 
    echo "Stop iteration reached. Terminating..."; 
    exit 1; 
fi

export PYTHONPATH=/work/pi_miyyer_umass_edu/rrajendhran/miniconda3/envs/llama3/bin/python
source /work/pi_miyyer_umass_edu/rrajendhran/miniconda3/etc/profile.d/conda.sh
conda activate llama3

wandb disabled
# mkdir /work/pi_miyyer_umass_edu/rrajendhran/huggingface_cache
export HF_HOME="/work/pi_miyyer_umass_edu/rrajendhran/huggingface_cache"
export HF_DATASETS_CACHE="/work/pi_miyyer_umass_edu/rrajendhran/huggingface_cache"
export TRANSFORMERS_CACHE="/work/pi_miyyer_umass_edu/rrajendhran/huggingface_cache"
module load cuda/12.4.0

export WANDB_API_KEY="9907271fbe6f6516b55140d83cc87a471d4835e9"

PREVITERATION=$((ITERATION-1))
OLDMODELNAME="finetuned_Llama3.1-Instruct_${PREVITERATION}"
MODELNAME="finetuned_Llama3.1-Instruct_${ITERATION}"
SAVEMODELNAME="Llama3.1-Instruct_FineTuned_${ITERATION}"
MODELPATH="/datasets/ai/llama3/meta-llama/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/5206a32e0bd3067aef1ce90f5528ade7d866253f/"
SAVEMODELPATH="./model_outputs/${SAVEMODELNAME}"
TRAINFILES="./inferenceOuts/${OLDMODELNAME}/${OLDMODELNAME}_prompts_train_commonsense_qa_correct.json ./inferenceOuts/${OLDMODELNAME}/${OLDMODELNAME}_prompts_train_commonsense_qa_rationalizedCorrect.json"
NEXTITERATION=$((ITERATION+1))
NEXTSTEPS=$(echo "$NUMSTEPS*1.2/1" | bc )

echo "**SCHEDULE JOBS**"
echo "ITERATION : ${ITERATION}"
echo "NUMSTEPS : ${NUMSTEPS}"
echo "STOPITERATION : ${STOPITERATION}"
echo "<<FINETUNE>>"
python3 finetune.py -trainFiles $TRAINFILES -model gptj -size 6b -modelPath $MODELPATH -dataset commonsense_qa -trainPattern .*/.*\.json -log stdout -batchSize 8 -numEpochs $NUMSTEPS -learningRate 1e-6 -savePath ./model/ -saveName $SAVEMODELNAME -maxSteps $NUMSTEPS -trainPrompt ./datasets/commonsense_qa/prompts.txt
if [ $? != 0 ];
then
    echo "exit 1"
fi
echo "<<INFERENCE>> [VALIDATION]"
deepspeed --num_gpus=1 inference.py -deepSpeed -out ./inferenceOuts/ -rationalize -trainPrompts -trainFiles ./datasets/commonsense_qa/prompts.txt -hintTrainPrompts ./datasets/commonsense_qa/promptsWithHints.txt -testFiles validation -model gptj -size 6b -dataset commonsense_qa -maxShots 9 -trainPattern .*/.*\.json -testPattern .*/.*\.json -log stdout -SAVEMODELPATH $SAVEMODELPATH -saveAs $MODELNAME
if [ $? != 0 ];
then
    echo "exit 1"
fi
echo "<<INFERENCE>> [TRAIN]"
deepspeed --num_gpus=1 inference.py -deepSpeed -out ./inferenceOuts/ -rationalize -trainPrompts -trainFiles ./datasets/commonsense_qa/prompts.txt -hintTrainPrompts ./datasets/commonsense_qa/promptsWithHints.txt -testFiles train -model gptj -size 6b -dataset commonsense_qa -maxShots 9 -trainPattern .*/.*\.json -testPattern .*/.*\.json -log stdout -SAVEMODELPATH $SAVEMODELPATH -saveAs $MODELNAME
if [ $? != 0 ];
then
    echo "exit 1"
fi
echo "<<NEXT>>"
sbatch scheduleJobs.sh -n $NEXTSTEPS -i $NEXTITERATION  -s $STOPITERATION