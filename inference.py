# STaRC - Self-Taught Reasoner with Certainty
import argparse
import wandb
import logging
import torch
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    AutoConfig,
    T5Tokenizer,
    LlamaForCausalLM,
)
from datasets import load_dataset, DatasetDict
import numpy as np
import json
import json
from tqdm import tqdm
import regex as re
import os
import glob
from os.path import exists
from pathlib import Path
from peft import PeftModel, PeftConfig

MODEL_TEMPERATURE = 0.1
MODEL_TOP_P = 0.9
MODEL_TOP_K = 0
MODEL_MAX_NEW_TOKENS = 256
MODEL_DO_SAMPLE = True
MODEL_REPETITION_PENALTY = 1.0
MAX_LENGTH = 4096

supportedModels = ["gptj", "unifiedqa", "llama3.1-instruct"]
supportedSizes = {"gptj": ["6b"], "unifiedqa": ["3b"], "llama3.1-instruct": ["8b"]}
supportedDatasets = ["commonsense_qa", "gsm8k", "arithmetic"]
supportedHFDatasets = ["commonsense_qa", "gsm8k"]

promptHeader = {
    "gptj": "",
    "unifiedqa": "",
    "llama3.1-instruct": """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful assistant<|eot_id|>""",
}

promptFormat = {
    "gptj": {
        "header": "",
        "content": "{input}{output}",
        "footer": "",
    },
    "unifiedqa": {
        "header": "",
        "content": "{input}{output}",
        "footer": "",
    },
    "llama3.1-instruct": {
        "header": """<|start_header_id|>user<|end_header_id|>

""",
        "content": """{input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{output}""",
        "footer": "<|eot_id|>",
    },
}

parser = argparse.ArgumentParser()

parser.add_argument(
    "-log", type=str, help="Path to file to print logging information", default=None
)

parser.add_argument("-cache_dir", type=str, help="Path to HF cache", required=True)

parser.add_argument(
    "-model",
    choices=supportedModels,
    help="Name of HuggingFace model to use",
    default="gptj",
)

parser.add_argument(
    "-modelPath", type=str, help="Path to (finetuned) model to use", default=None
)

parser.add_argument("-size", help="Size of HuggingFace model to use", default="6b")

parser.add_argument(
    "-dataset",
    choices=supportedDatasets,
    help="Name of HuggingFace dataset to use",
    default="commonsense_qa",
)

parser.add_argument(
    "-direct", action="store_true", help="Boolean flag to enable direct prompting"
)

parser.add_argument(
    "-maxShots",
    type=int,
    help="Maximum no. of shots to use in few-shot setting",
    default=9,
)

parser.add_argument(
    "-zeroShot", action="store_true", help="Boolean flag to enable zero shot evaluation"
)

parser.add_argument(
    "-trainFiles",
    nargs="+",
    help="List of paths to json/txt files containing training data/prompts",
    default=["train"],
)

parser.add_argument(
    "-trainPrompts",
    action="store_true",
    help="Boolean flag to indicate that -trainFiles points to files containing train prompts",
)

parser.add_argument(
    "-hintTrainPrompts",
    nargs="+",
    help="List of paths to txt files containing training prompts with hints",
)

parser.add_argument(
    "-testFiles",
    nargs="+",
    help="List of paths to json files containing test data",
    default=["validation"],
)

parser.add_argument(
    "-isTrainDir",
    action="store_true",
    help="Boolean flag to indicate if the -trainFiles input is a directory path",
)

parser.add_argument(
    "-isTestDir",
    action="store_true",
    help="Boolean flag to indicate if the -testFiles input is a directory path",
)

parser.add_argument(
    "-trainPattern",
    help="RegEx pattern for json/txt file names in the train directory that need to be used",
)

parser.add_argument(
    "-testPattern",
    help="RegEx pattern for json file names in the test directory that need to be merged",
)

parser.add_argument(
    "-out",
    help="Path to directory where outputs are to be saved",
    default="./inferenceOuts/",
)

parser.add_argument("-saveAs", help="Prefix to add to output files", default="")

parser.add_argument(
    "-rationalize", action="store_true", help="Boolean flag to enable rationalization"
)

parser.add_argument(
    "-uncertainty",
    action="store_true",
    help="Boolean flag to enable uncertainty computation",
)

parser.add_argument(
    "-method",
    type=str,
    choices=["ppl"],
    help="Type of uncertainty method to use",
    default="ppl",
)

parser.add_argument(
    "-methodParam",
    help="Parameter for uncertainty method",
)


# ---------------------------------------------------------------------------
def _generateIndividualPrompt(
    instance, dataset, model, direct=False, rationalize=False, isTest=False
):
    if dataset not in supportedDatasets:
        raise ValueError(f"{dataset} not supported!")

    inp, out = "", ""

    # commonsense_qa on HuggingFace
    # {
    #     "id": (string),
    #     "question": (string),
    #     "choices": {
    #         "labels": [(string),...],
    #         "text": [(string),...]
    #     },
    #     "answerKey": (string)
    # }
    if dataset == "commonsense_qa":
        if not direct and not isTest:
            raise ValueError(
                "Only direct prompting supported with commonsense_qa dataset on HuggingFace!"
            )
        inp += "Q: " + instance["question"] + "\nAnswer Choices:\n"
        for c, t in zip(instance["choices"]["label"], instance["choices"]["text"]):
            inp += "({}) {}".format(c.lower(), t.lower())
            if rationalize:
                if c == instance["answerKey"]:
                    inp += " (CORRECT)"
            inp += "\n"
        inp += "A: "
        if not isTest:
            out += "({}).\n\n".format(instance["answerKey"].lower())
    # gsm8k on HuggingFace
    # {
    #     "question": (string),
    #     "answer": (string)
    # }
    elif dataset == "gsm8k":
        inp += "Q: " + instance["question"]
        extractedAnswer = extractAnswer(instance["answer"], dataset, direct)
        if rationalize:
            inp += " ({})".format(extractedAnswer["answer"])
        inp += "\nA: "
        if not isTest:
            if direct:
                out += extractedAnswer["answer"] + "\n\n"
            else:
                out += instance["answer"] + "\n\n"
    # arithmetic (Not on Huggingface)
    # {
    #     "question": (string),
    #     "answer": (string),
    #     "scratch": (string), [OPTIONAL]
    # }
    elif dataset == "arithmetic":
        inp += "Input:\n" + instance["question"].strip()
        inp += "\nTarget:\n"
        if rationalize:
            inp += instance["answer"] + "\n\n"
        if not isTest:
            if not direct:
                out += instance["scratch"].strip() + "\n"
            out += instance["answer"] + "\n\n"
    if isTest:
        return promptFormat[model]["header"] + promptFormat[model]["content"].format(
            input=inp, output=""
        )
    return (
        promptFormat[model]["header"]
        + promptFormat[model]["content"].format(input=inp, output=out)
        + promptFormat[model]["footer"]
    )


# ---------------------------------------------------------------------------
def _generatePrompt(
    data, dataset, model, maxShots, direct=False, rationalize=False, isTest=False
):
    prompts = []

    if dataset not in supportedDatasets:
        raise ValueError(f"{dataset} not supported!")

    for index, instance in enumerate(data):
        if index >= maxShots:
            break
        prompts.append(
            _generateIndividualPrompt(
                instance, dataset, model, direct, rationalize, isTest
            )
        )

    return prompts


# ---------------------------------------------------------------------------
def generateTrainPrompt(data, dataset, model, maxShots, direct, rationalize=False):
    return "".join(
        _generatePrompt(data, dataset, model, maxShots, direct, rationalize, False)
    )


# ---------------------------------------------------------------------------
def generateTestPrompt(instance, dataset, model, maxShots, direct, rationalize=False):
    return "".join(
        _generatePrompt([instance], dataset, model, maxShots, direct, rationalize, True)
    )


# ---------------------------------------------------------------------------
def extractAnswer(answer, dataset, direct=False):
    """
    Extracts the answer and optional rationale from a model-generated response based on 
    the specified dataset format.

    Parameters:
    -----------
    answer (str): The model-generated text containing an answer.
    dataset (str): Name of the dataset for which the answer extraction is performed. Supports "commonsense_qa", "gsm8k", and "arithmetic".
    direct (bool, optional, default=False): If True, applies a more direct extraction method, assuming answer format without additional context or rationale.

    Returns:
    --------
    dict or None: A dictionary containing the extracted answer and, if available, the rationale. If extraction fails, returns None.

    Example:
    --------
    result = extractAnswer("The answer is (b).", "commonsense_qa")
    print(result)
    {'answer': 'b'}

    result = extractAnswer("#### 42", "gsm8k", direct=True)
    print(result)
    {'answer': '42'}
    """
    if dataset not in supportedDatasets:
        raise ValueError(f"{dataset} not supported!")
    if dataset == "commonsense_qa":
        if not direct:
            searchPattern = "answer is .*."
        else:
            searchPattern = "\([a-z]\)."
        matchedSpan = re.search(searchPattern, answer)
        if matchedSpan == None:
            logging.warning(f"Could not extract answer from {answer}!")
            return None
            # raise RuntimeError(f"Could not extract answer from {answer}!")
        extractedAnswer = answer[matchedSpan.start() : matchedSpan.end()].strip()
        answerPattern = "\([a-z]\)."
        matchedAnswer = re.findall(answerPattern, extractedAnswer)
        if len(matchedAnswer) == 0:
            logging.warning(f"Could not extract answer from {extractedAnswer}!")
            return None
            # raise RuntimeError(f"Could not extract answer from {extractedAnswer}!")
        matchedAnswer = matchedAnswer[-1][1]
        extractedAnswer = {
            "answer": matchedAnswer.strip(),
        }
        if not direct:
            rationale = answer[: matchedSpan.start()]
            rationalePattern = "[.]"
            matchedRationale = re.split(rationalePattern, rationale)
            if len(matchedRationale):
                rationale = ".".join(matchedRationale[:-1]) + "."
            extractedAnswer.update(
                {
                    "rationale": rationale.strip(),
                }
            )
    elif dataset == "gsm8k":
        if not direct:
            searchPattern = "\n#### [0-9]+"
        else:
            searchPattern = "[0-9]+"
        matchedSpan = re.search(searchPattern, answer)
        if matchedSpan == None:
            logging.warning(f"Could not extract answer from {answer}!")
            return None
            # raise RuntimeError(f"Could not extract answer from {answer}!")
        extractedAnswer = answer[matchedSpan.start() : matchedSpan.end()].strip()
        if not direct:
            matchedAnswer = re.sub("#", "", extractedAnswer).strip()
        else:
            matchedAnswer = extractedAnswer.strip()
        extractedAnswer = {
            "answer": matchedAnswer.strip(),
        }
        if not direct:
            rationale = answer[: matchedSpan.start()]
            extractedAnswer.update(
                {
                    "rationale": rationale.strip(),
                }
            )
    elif dataset == "arithmetic":
        if not direct:
            searchPattern = "</scratch>\n"
        else:
            searchPattern = "Target:\n"
        matchedSpan = re.search(searchPattern, answer)
        if matchedSpan == None:
            logging.warning(f"Could not extract answer from {answer}!")
            return None
            # raise RuntimeError(f"Could not extract answer from {answer}!")
        matchedAnswer = answer[matchedSpan.end() :].strip()
        if "\n" in matchedAnswer:
            matchedAnswer = matchedAnswer[: matchedAnswer.index("\n")]
        extractedAnswer = {
            "answer": matchedAnswer.strip(),
        }
        if not direct:
            scratchStart = "<scratch>"
            scratchEnd = "</scratch>\n"
            matchedStartSpan = re.search(scratchStart, answer)
            matchedEndSpan = re.search(scratchEnd, answer)
            scratch = answer[matchedStartSpan.start() : matchedEndSpan.end()]
            extractedAnswer.update(
                {
                    "rationale": scratch.strip(),
                }
            )
    return extractedAnswer


# ---------------------------------------------------------------------------
def processArguments(args):
    config = {
        "logFile": args.log,
        "model": args.model,
        "modelPath": args.modelPath,
        "size": args.size,
        "dataset": args.dataset,
        "direct": args.direct,
        "maxShots": args.maxShots,
        "zeroShot": args.zeroShot,
        "trainFiles": args.trainFiles,
        "trainPrompts": args.trainPrompts,
        "isTrainDir": args.isTrainDir,
        "trainPattern": args.trainPattern,
        "testFiles": args.testFiles,
        "isTestDir": args.isTestDir,
        "testPattern": args.testPattern,
        "outPath": args.out,
        "saveAs": args.saveAs,
        "rationalize": args.rationalize,
        "hintTrainPrompts": args.hintTrainPrompts,
        "uncertainty": args.uncertainty,
        "method": args.method,
        "methodParam": args.methodParam,
    }

    if config["logFile"]:
        if config["logFile"] != "stdout" and config["logFile"].endswith(".txt"):
            logging.basicConfig(
                filename=config["logFile"], filemode="w", level=logging.INFO
            )
        elif config["logFile"] == "stdout":
            logging.basicConfig(filemode="w", level=logging.INFO)
        elif config["logFile"] == "none":
            logging.basicConfig(filemode="w", level=logging.ERROR)
        else:
            raise ValueError("Invalid log file {}!".format(config["logFile"]))
    else:
        logging.basicConfig(filemode="w", level=logging.ERROR)

    if config["direct"] and config["rationalize"]:
        raise ValueError("Cannot perform rationalization in direct prompting mode!")
    if config["rationalize"] and config["trainPrompts"]:
        if not config["hintTrainPrompts"]:
            raise ValueError(
                "Hint train prompts must be provided with the hintTrainPrompts flag when these flags are set: trainPrompts, rationalize"
            )
        if len(config["trainFiles"]) != len(config["hintTrainPrompts"]):
            raise ValueError(
                "Every train file must have a hinted version when these flags are set: trainPrompts, rationalize"
            )

    if config["isTrainDir"]:
        jsonDirName = config["trainFiles"][0]
        jsonPattern = os.path.join(jsonDirName, "*.json")
        config["trainFiles"] = glob.glob(jsonPattern)
        if config["trainPattern"]:
            try:
                re.compile(config["trainPattern"])
            except:
                raise ValueError(
                    "{} is not a valid regular expression!".format(config["trainPattern"])
                )
            config["trainFiles"] = [
                tf for tf in config["trainFiles"] if re.match(config["trainPattern"], tf)
            ]
            if len(config["trainFiles"]) == 0:
                raise RuntimeError(
                    "{} did not match any file!".format(config["trainPattern"])
                )

    if config["isTestDir"]:
        jsonDirName = config["testFiles"][0]
        jsonPattern = os.path.join(jsonDirName, "*.json")
        config["testFiles"] = glob.glob(jsonPattern)
        if config["testPattern"]:
            try:
                re.compile(config["testPattern"])
            except:
                raise ValueError(
                    "{} is not a valid regular expression!".format(config["testPattern"])
                )
            config["testFiles"] = [
                tf for tf in config["testFiles"] if re.match(config["testPattern"], tf)
            ]
            if len(config["testFiles"]) == 0:
                raise RuntimeError(
                    "{} did not match any file!".format(config["testPattern"])
                )

    # Check if file exists
    for trainFile in config["trainFiles"]:
        if (
            not trainFile.endswith(".json")
            and not trainFile.endswith(".jsonl")
            and not trainFile.endswith(".txt")
        ):
            logging.warning(
                f"Train File '{trainFile}' is not a json/jsonl/txt file. Not checking if file exists. Ignore this warning if this is expected behaviour."
            )
            continue
        file_exists = exists(trainFile)
        if not file_exists:
            raise ValueError(f"{trainFile} is an invalid train file path!")
        path = Path(trainFile)
        if not path.is_file():
            raise ValueError(f"{trainFile} is not a (train) file!")
    # Check if file exists
    for testFile in config["testFiles"]:
        if not testFile.endswith(".json") and not testFile.endswith(".jsonl"):
            logging.warning(
                f"Test File '{testFile}' is not a json/jsonl file. Not checking if file exists. Ignore this warning if this is expected behaviour."
            )
            continue
        file_exists = exists(testFile)
        if not file_exists:
            raise ValueError(f"{testFile} is an invalid test file path!")
        path = Path(testFile)
        if not path.is_file():
            raise ValueError(f"{testFile} is not a (test) file!")

    return config


# ---------------------------------------------------------------------------
def infer(model, modelName, tokenizer, prompt, generationConfig={}):
    """
    Generates text from a model based on a provided prompt and optional generation 
    configuration settings.

    Parameters:
    -----------
    model (torch.nn.Module): Model to use
    modelName (str): Name of the model
    tokenizer (transformers.PreTrainedTokenizer): The tokenizer used
    prompt (str): The input text prompt
    generationConfig (dict, optional, default={}): A dict of params for text generation

    Returns:
    --------
    str: The generated text based on the input prompt.

    Example:
    --------
    generated_text = infer(model, "gptj", tokenizer, "Once upon a time")
    print(generated_text)
    "Once upon a time, in a distant land..."

    Notes:
    ------
    - `MODEL_MAX_NEW_TOKENS` defines the maximum number of new tokens generated per call.
    - The `pad_token_id` is set to the model's EOS token to avoid padding issues.
    """

    tokenizedInput = tokenizer(prompt, return_tensors="pt")
    inputIDs = tokenizedInput.input_ids.to(device=model.device)
    attentionMask = tokenizedInput.attention_mask.to(device=model.device)

    print(prompt)
    try:
        genTokens = model.generate(
            input_ids=inputIDs,
            attention_mask=attentionMask,
            max_new_tokens=MODEL_MAX_NEW_TOKENS,
            pad_token_id=tokenizer.eos_token_id,
            **generationConfig,
        )

        if modelName == "gptj" or modelName == "llama3.1-instruct":
            outputIDs = genTokens[0, len(inputIDs[0]) :]
        else:
            outputIDs = genTokens[0, :]
        genText = tokenizer.decode(outputIDs)
        return genText
    except Exception as e:
        print(f"An error occurred during inference: {e}")
        return ""


# ---------------------------------------------------------------------------
def compute_uncertainty(model, modelName, tokenizer, prompt, response, method="ppl"):
    """
    Computes an uncertainty metric for a model's response to a given prompt using 
    the specified uncertainty quantification method. 

    Parameters:
    -----------
    model : PreTrainedModel
        The language model to evaluate.
    modelName : str
        Name of the model, which determines how input and target tokens are set 
        (supported values: "gptj", "llama3.1-instruct", "unifiedqa").
    tokenizer : PreTrainedTokenizer
        Tokenizer associated with the model, used to convert text into token IDs.
    prompt : str
        The input prompt or context provided to the model.
    response : str
        The model-generated response to the prompt.
    method : str, optional (default="ppl")
        The uncertainty quantification method. 

    Returns:
    --------
    float
        The computed perplexity (PPL) score as a measure of uncertainty, where a 
        lower perplexity indicates a higher confidence in the response.

    Raises:
    -------
    ValueError: If `modelName` is not recognized or if `method` is not "ppl".
        
    Notes:
    ------
    - For "gptj" and "llama3.1-instruct" models:
        - The function evaluates the model's performance on the `response` section 
          of the concatenated `prompt + response` text, ignoring the prompt tokens 
          in the loss calculation by setting their target token values to -100.
    - For "unifiedqa" model:
        - The function treats the `prompt` and `response` separately, and evaluates 
          the model solely on its ability to predict `response`.
    
    Example:
    --------
    ppl = compute_uncertainty(model, "gptj", tokenizer, "What is AI?", "AI is the simulation of human intelligence in machines.")
    print(f"Perplexity: {ppl}")
    """

    if method == "ppl":
        if modelName == "gptj" or modelName == "llama3.1-instruct":
            input_ids = tokenizer.encode(
                prompt + response, padding="longest", truncation=True, return_tensors="pt"
            )
            response_ids = tokenizer.encode(
                response, padding="longest", truncation=True, return_tensors="pt"
            )
            target_ids = input_ids.clone()
            target_ids[0, : -response_ids.shape[-1]] = -100
        elif modelName == "unifiedqa":
            input_ids = tokenizer.encode(
                prompt, padding="longest", truncation=True, return_tensors="pt"
            )
            response_ids = tokenizer.encode(
                response, padding="longest", truncation=True, return_tensors="pt"
            )
            target_ids = response_ids
        else:
            raise ValueError("Unrecognized model: {}".format(modelName))
        outputs = model(input_ids=input_ids, labels=target_ids)
        ppl = np.exp(outputs.loss.item())
        return ppl
    else:
        raise ValueError(
            "Unrecognized uncertainty quantification method: {}".format(method)
        )


# ---------------------------------------------------------------------------
def is_uncertain(uncertainty, method_param, method="ppl"):
    if method == "ppl":
        if uncertainty > float(method_param):
            return True
        return False
    else:
        raise ValueError(
            "Unrecognized uncertainty quantification method: {}".format(method)
        )


# ---------------------------------------------------------------------------
def main():
    """
    Main function to load, configure, run inference, and evaluate a language model using Weights & Biases for experiment tracking.
    Supports various model architectures and dataset configurations for making predictions and optionally performs rationalization.

    Function Workflow:
    -----------
        1. Parse command-line arguments to configure the script.
        2. Initialize a Weights & Biases project for experiment tracking.
        3. Set the computation device based on availability (CPU or GPU).
        4. Load a specified pre-trained or fine-tuned model and tokenizer from Hugging Face.
        5. Check for dataset availability and load it. Split the data into training and testing sets, if applicable.
        6. Iterate over specified training and testing datasets to generate predictions.
        7. For each example, generate a prompt and run inference to get model predictions.
        8. Optionally, perform rationalization: attempt to correct wrong predictions by re-formulating the problem.
        9. Log each step of the process, including detailed information about inputs, predictions, and correct answers.
        10. Save results for correct and incorrect predictions, both before and after rationalization, in JSON files.
        11. Use Weights & Biases to finish tracking.

    Exceptions:
    -----------
        ValueError: Thrown for unsupported model sizes, datasets, or when required configuration files are missing.
        NotImplementedError: Raised for unsupported datasets or methods if not implemented.
    """
    args = parser.parse_args()

    wandb.init(project="STaRC", config=processArguments(args))

    config = wandb.config

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    if config.model == "gptj":
        if config.size == "6b":
            modelID = "EleutherAI/gpt-j-6B"
            if config.modelPath and config.modelPath != modelID:  # Finetuned model
                print(
                    "Using finetuned (PEFT) model and tokenizer from {}".format(
                        config.modelPath
                    )
                )
                peftConfig = PeftConfig.from_pretrained(config.modelPath)
                model = AutoModelForCausalLM.from_pretrained(
                    peftConfig.base_model_name_or_path
                )
                model = PeftModel.from_pretrained(model, config.modelPath)
                # tokenizer = AutoTokenizer.from_pretrained(peftConfig.base_model_name_or_path)
                tokenizer = AutoTokenizer.from_pretrained(config.modelPath)
                generationConfig = {}
            else:  # Pretrained model
                print(
                    "Using pretrained model and tokenizer from {} on HuggingFace".format(
                        modelID
                    )
                )
                model = AutoModelForCausalLM.from_pretrained(modelID, device_map="auto")
                tokenizer = AutoTokenizer.from_pretrained(modelID)
                generationConfig = {
                    # "do_sample":MODEL_DO_SAMPLE,
                    # "temperature":MODEL_TEMPERATURE,
                    # "top_p":MODEL_TOP_P,
                    # "top_k":MODEL_TOP_K,
                    # "repetition_penalty":MODEL_REPETITION_PENALTY,
                }
        else:
            raise ValueError(
                "Only {} size(s) supported!".format(
                    "/".join(supportedSizes[config.model])
                )
            )
    elif config.model == "unifiedqa":
        if config.size == "3b":
            modelID = "allenai/unifiedqa-t5-3b"
            if config.modelPath and config.modelPath != modelID:  # Finetuned model
                print(
                    "Using finetuned (PEFT) model and tokenizer from {}".format(
                        config.modelPath
                    )
                )
                peftConfig = PeftConfig.from_pretrained(config.modelPath)
                model = AutoModelForSeq2SeqLM.from_pretrained(
                    peftConfig.base_model_name_or_path
                )
                model = PeftModel.from_pretrained(model, config.modelPath)
                # tokenizer = AutoTokenizer.from_pretrained(peftConfig.base_model_name_or_path)
                tokenizer = T5Tokenizer.from_pretrained(config.modelPath)
                generationConfig = {}
            else:  # Pretrained model
                print(
                    "Using pretrained model and tokenizer from {} on HuggingFace".format(
                        modelID
                    )
                )
                model = AutoModelForSeq2SeqLM.from_pretrained(modelID, device_map="auto")
                tokenizer = T5Tokenizer.from_pretrained(modelID)
                generationConfig = {
                    # "do_sample":MODEL_DO_SAMPLE,
                    # "temperature":MODEL_TEMPERATURE,
                    # "top_p":MODEL_TOP_P,
                    # "top_k":MODEL_TOP_K,
                    # "repetition_penalty":MODEL_REPETITION_PENALTY,
                }
        else:
            raise ValueError(
                "Only {} size(s) supported!".format(
                    "/".join(supportedSizes[config.model])
                )
            )
    elif config.model == "llama3.1-instruct":
        if config.size == "8b":
            modelID = "meta-llama/Llama-3.1-8B"
            if config.modelPath != modelID:
                modelID = config.modelPath
            model = LlamaForCausalLM.from_pretrained(
                modelID,
                torch_dtype=torch.bfloat16,
                device_map="balanced",
                cache_dir=args.cache_dir,
                # attn_implementation="flash_attention_2",
            )

            tokenizer = AutoTokenizer.from_pretrained(modelID, cache_dir=args.cache_dir)
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            model.generation_config.pad_token_id = tokenizer.pad_token_id
            model.resize_token_embeddings(len(tokenizer))
            generationConfig = {
                # "do_sample":MODEL_DO_SAMPLE,
                # "temperature":MODEL_TEMPERATURE,
                # "top_p":MODEL_TOP_P,
                # "top_k":MODEL_TOP_K,
                # "repetition_penalty":MODEL_REPETITION_PENALTY,
            }
        else:
            raise ValueError(
                "Only {} size(s) supported!".format(
                    "/".join(supportedSizes[config.model])
                )
            )
    else:
        raise ValueError("Only {} model(s) supported!".format("/".join(supportedModels)))

    if model.dtype == torch.float32:
        model.to(device=device)

    if config.dataset not in supportedDatasets:
        raise ValueError(
            "Only {} dataset(s) supported!".format("/".join(supportedDatasets))
        )

    if config.dataset in supportedHFDatasets:
        if config.dataset == "gsm8k":
            dataset_train = load_dataset('gsm8k', "main", split="train")
            dataset_test = load_dataset('gsm8k', "main", split="test")

            print(f"Train size: {len(dataset_train)}")
            print(f"Test size: {len(dataset_test)}")

            dataset = DatasetDict({"train": dataset_train, "test": dataset_test})
        elif config.dataset == "commonsense_qa":
            dataset_train = load_dataset('commonsense_qa' , split="train[0:1000]")
            dataset_valid = load_dataset('commonsense_qa', split="validation")
            dataset_test = load_dataset('commonsense_qa', split="test")

            print(f"Train size: {len(dataset_train)}")
            print(f"Valid size: {len(dataset_train)}")
            print(f"Test size: {len(dataset_test)}")

            dataset = DatasetDict({
                'train': dataset_train,
                'validation': dataset_valid,
                'test': dataset_test
            })
        else:
            dataset = load_dataset(config.dataset)

    print(f"Model: {config.model}-{config.size}")
    print(f"Model Path: {config.modelPath}")
    print(f"Dataset: {config.dataset}")
    if config.rationalize:
        print("Performing rationalization")
    else:
        print("Not performing rationalization")

    with torch.no_grad():
        for trainInd, trainFile in enumerate(tqdm(config.trainFiles, desc="Train File")):
            print("Train file: {}".format(trainFile))
            if not config.zeroShot:
                if trainFile.endswith(".json"):
                    with open(trainFile, "r") as f:
                        trainData = json.load(f)
                elif trainFile.endswith(".jsonl"):
                    trainData = []
                    with open(trainFile, "r") as f:
                        for line in f:
                            trainData.append(json.loads(line))
                elif config.trainPrompts:
                    if trainFile.endswith(".txt"):
                        with open(trainFile, "r") as f:
                            trainPrompt = f.read()
                    else:
                        raise ValueError(
                            f"{trainFile} prompt file does not have .txt extension!"
                        )
                    if config.rationalize:
                        hintTrainFile = config.hintTrainPrompts[trainInd]
                        if hintTrainFile.endswith(".txt"):
                            with open(hintTrainFile, "r") as f:
                                rationalizedTrainPrompt = f.read()
                        else:
                            raise ValueError(
                                f"{hintTrainFile} prompt file with hints does not have .txt extension!"
                            )
                elif config.dataset in supportedHFDatasets:
                    trainData = list(
                        dataset[trainFile].select(
                            np.random.choice(len(dataset[trainFile]), config.maxShots)
                        )
                    )
                else:
                    raise ValueError(
                        f"Neither is {config.dataset} on HugginFace nor has path to files containing training data been provided!"
                    )
                if not config.trainPrompts:
                    trainPrompt = generateTrainPrompt(
                        trainData,
                        config.dataset,
                        config.model,
                        config.maxShots,
                        config.direct,
                        False,
                    )
                    rationalizedTrainPrompt = generateTrainPrompt(
                        trainData,
                        config.dataset,
                        config.model,
                        config.maxShots,
                        config.direct,
                        True,
                    )

            for testInd, testFile in enumerate(tqdm(config.testFiles, desc="Test File")):
                print("Test file: {}".format(testFile))
                model.eval()
                if config.dataset in supportedDatasets:
                    if testFile.endswith(".jsonl"):
                        testData = []
                        with open(testFile, "r") as f:
                            for line in f:
                                testData.append(json.loads(line))
                    elif testFile.endswith(".json"):
                        with open(testFile, "r") as f:
                            testData = json.load(f)
                        if config.dataset == "arithmetic":
                            # Randomly sample 10000 examples
                            testData = np.array(testData)[
                                np.random.choice(len(testData), 10000)
                            ]
                    elif config.dataset in supportedHFDatasets:
                        testData = list(dataset[testFile])
                    else:
                        raise ValueError(
                            "Test File {} does not have json/jsonl extension and is also not on Huggingface!".format(
                                testFile
                            )
                        )
                else:
                    raise ValueError(
                        "{} not supported for testing!".format(config.dataset)
                    )
                outputs = []
                correctPreds = []
                wrongPreds = []
                rationalizedCorrectPreds = []
                rationalizedWrongPreds = []
                accuracyScore = 0
                rationalizedAccuracyScore = 0
                for testInstance in tqdm(testData, desc="Test Instance"):
                    testPrompt = generateTestPrompt(
                        testInstance,
                        config.dataset,
                        config.model,
                        config.maxShots,
                        config.direct,
                        False,
                    )

                    if not config.zeroShot:
                        finalPrompt = (
                            promptHeader[config.model] + trainPrompt + testPrompt
                        )
                    else:
                        finalPrompt = promptHeader[config.model] + testPrompt

                    genText = infer(
                        model, config.model, tokenizer, finalPrompt, generationConfig
                    )

                    extractedAnswer = extractAnswer(
                        genText, config.dataset, config.direct
                    )
                    if extractedAnswer == None:
                        continue
                    prediction = extractedAnswer["answer"]
                    testInstance.update(
                        {
                            "output": genText,
                            "prediction": prediction,
                        }
                    )

                    if not config.direct:
                        rationale = extractedAnswer["rationale"]
                        testInstance.update(
                            {
                                "rationale": rationale,
                            }
                        )

                    logging.info(f"Prompt:\n{finalPrompt}")
                    if not config.direct:
                        logging.info(f"Rationale: {rationale}")
                    # logging.info(f"Model Output: {genText}")
                    if config.dataset == "commonsense_qa":
                        logging.info(f"Prediction: {prediction}")
                        logging.info(
                            "Answer: {}".format(testInstance["answerKey"].lower())
                        )
                        logging.info(
                            "Score: {}".format(
                                prediction.lower() == testInstance["answerKey"].lower()
                            )
                        )
                    elif config.dataset == "gsm8k":
                        corrAnswer = extractAnswer(
                            testInstance["answer"], config.dataset, config.direct
                        )
                        if corrAnswer == None:
                            continue
                        if not config.direct:
                            logging.info(
                                "Gold Rationale: {}".format(corrAnswer["rationale"])
                            )
                        logging.info(f"Prediction: {prediction}")
                        logging.info("Answer: {}".format(corrAnswer["answer"].lower()))
                        logging.info(
                            "Score: {}".format(
                                prediction.lower() == corrAnswer["answer"].lower()
                            )
                        )
                    elif config.dataset == "arithmetic":
                        if not config.direct:
                            logging.info(
                                "Gold Rationale: {}".format(testInstance["scratch"])
                            )
                        logging.info(f"Prediction: {prediction}")
                        logging.info(
                            "Answer: {}".format(testInstance["answerKey"].lower())
                        )
                        logging.info(
                            "Score: {}".format(
                                prediction.lower() == testInstance["answerKey"].lower()
                            )
                        )

                    logging.info("-" * 25)

                    outputs.append(testInstance)
                    curInstanceCorrect = False
                    if config.dataset == "commonsense_qa":
                        if prediction.lower() == testInstance["answerKey"].lower():
                            accuracyScore += 1
                            correctPreds.append(testInstance)
                            curInstanceCorrect = True
                    elif config.dataset == "gsm8k":
                        if prediction.lower() == corrAnswer["answer"].lower():
                            accuracyScore += 1
                            correctPreds.append(testInstance)
                            curInstanceCorrect = True
                    elif config.dataset == "arithmetic":
                        if prediction.lower() == testInstance["answerKey"].lower():
                            accuracyScore += 1
                            correctPreds.append(testInstance)
                            curInstanceCorrect = True

                    if config.uncertainty:
                        uncertainty = compute_uncertainty(
                            model,
                            config.model,
                            tokenizer,
                            finalPrompt,
                            rationale,
                            config.method,
                        )
                        model_uncertain = is_uncertain(
                            uncertainty, config.methodParam, config.method
                        )
                        curInstanceCorrect = model_uncertain
                        wandb.log(
                            {
                                "testInstance": testInstance,
                                "uncertainty": uncertainty,
                                "is_uncertain": model_uncertain,
                            }
                        )

                    if not curInstanceCorrect:
                        wrongPreds.append(testInstance)
                        # Rationalize
                        if config.rationalize:
                            rationalizedTestPrompt = generateTestPrompt(
                                testInstance,
                                config.dataset,
                                config.model,
                                config.maxShots,
                                config.direct,
                                True,
                            )

                            if not config.zeroShot:
                                rationalizedFinalPrompt = (
                                    rationalizedTrainPrompt + rationalizedTestPrompt
                                )
                            else:
                                rationalizedFinalPrompt = rationalizedTestPrompt
                            genText = infer(
                                model,
                                config.model,
                                tokenizer,
                                rationalizedFinalPrompt,
                                generationConfig,
                            )
                            extractedAnswer = extractAnswer(
                                genText, config.dataset, config.direct
                            )
                            if extractedAnswer == None:
                                continue
                            prediction = extractedAnswer["answer"]
                            testInstance.update(
                                {
                                    "output": genText,
                                    "prediction": prediction,
                                }
                            )

                            if not config.direct:
                                rationale = extractedAnswer["rationale"]
                                testInstance.update(
                                    {
                                        "rationale": rationale,
                                    }
                                )

                            logging.info("Performing rationalization...")
                            logging.info(
                                f"Rationalized Prompt:\n{rationalizedFinalPrompt}"
                            )
                            if not config.direct:
                                logging.info(f"Rationalized Rationale: {rationale}")
                            if config.dataset == "commonsense_qa":
                                logging.info(f"Rationalized Prediction: {prediction}")
                                logging.info(
                                    "Rationalized Answer: {}".format(
                                        testInstance["answerKey"].lower()
                                    )
                                )
                                logging.info(
                                    "Rationalized Score: {}".format(
                                        prediction.lower()
                                        == testInstance["answerKey"].lower()
                                    )
                                )
                            elif config.dataset == "gsm8k":
                                if not config.direct:
                                    logging.info(
                                        "Rationalized Gold Rationale: {}".format(
                                            corrAnswer["rationale"]
                                        )
                                    )
                                logging.info(f"Rationalized Prediction: {prediction}")
                                logging.info(
                                    "Rationalized Answer: {}".format(
                                        corrAnswer["answer"].lower()
                                    )
                                )
                                logging.info(
                                    "Rationalized Score: {}".format(
                                        prediction.lower() == corrAnswer["answer"].lower()
                                    )
                                )
                            elif config.dataset == "arithmetic":
                                if not config.direct:
                                    logging.info(
                                        "Rationalized Gold Rationale: {}".format(
                                            testInstance["scratch"]
                                        )
                                    )
                                logging.info(f"Rationalized Prediction: {prediction}")
                                logging.info(
                                    "Rationalized Answer: {}".format(
                                        testInstance["answerKey"].lower()
                                    )
                                )
                                logging.info(
                                    "Rationalized Score: {}".format(
                                        prediction.lower()
                                        == testInstance["answerKey"].lower()
                                    )
                                )
                            logging.info("-" * 25)

                            outputs.append(testInstance)
                            if config.dataset == "commonsense_qa":
                                if (
                                    prediction.lower()
                                    == testInstance["answerKey"].lower()
                                ):
                                    rationalizedAccuracyScore += 1
                                    rationalizedCorrectPreds.append(testInstance)
                                    curInstanceCorrect = True
                            elif config.dataset == "gsm8k":
                                if prediction.lower() == corrAnswer["answer"].lower():
                                    rationalizedAccuracyScore += 1
                                    rationalizedCorrectPreds.append(testInstance)
                                    curInstanceCorrect = True
                            elif config.dataset == "arithmetic":
                                if (
                                    prediction.lower()
                                    == testInstance["answerKey"].lower()
                                ):
                                    rationalizedAccuracyScore += 1
                                    rationalizedCorrectPreds.append(testInstance)
                                    curInstanceCorrect = True

                            if not curInstanceCorrect:
                                rationalizedWrongPreds.append(testInstance)
                    logging.info("*" * 50)
                #     break
                print("Accuracy: {:0.2f}% ({}/{})".format((accuracyScore/len(testData))*100, accuracyScore, len(testData)))
                if config.rationalize:
                    print("Rationalization Accuracy: {:0.2f}% ({}/{})".format((rationalizedAccuracyScore/(len(rationalizedCorrectPreds)+len(rationalizedWrongPreds)))*100, rationalizedAccuracyScore, (len(rationalizedCorrectPreds)+len(rationalizedWrongPreds))))
                if not config.outPath.endswith("/"):
                    config.outPath += "/"
                if not os.path.exists(f"{config.outPath}{config.saveAs}"):
                    os.makedirs(f"{config.outPath}{config.saveAs}")
                logging.info(
                    f"Saving inference outputs at {config.outPath}{config.saveAs}"
                )
                with open(
                    f'{config.outPath}{config.saveAs}/{trainFile.split("/")[-1].split(".")[0]}_{testFile.split("/")[-1].split(".")[0]}_{config.dataset}.json',
                    "w",
                ) as fout:
                    json.dump(outputs, fout)
                with open(
                    f'{config.outPath}{config.saveAs}/{trainFile.split("/")[-1].split(".")[0]}_{testFile.split("/")[-1].split(".")[0]}_{config.dataset}_correct.json',
                    "w",
                ) as fout:
                    json.dump(correctPreds, fout)
                with open(
                    f'{config.outPath}{config.saveAs}/{trainFile.split("/")[-1].split(".")[0]}_{testFile.split("/")[-1].split(".")[0]}_{config.dataset}_wrong.json',
                    "w",
                ) as fout:
                    json.dump(wrongPreds, fout)
                with open(
                    f'{config.outPath}{config.saveAs}/{trainFile.split("/")[-1].split(".")[0]}_{testFile.split("/")[-1].split(".")[0]}_{config.dataset}_rationalizedCorrect.json',
                    "w",
                ) as fout:
                    json.dump(rationalizedCorrectPreds, fout)
                with open(
                    f'{config.outPath}{config.saveAs}/{trainFile.split("/")[-1].split(".")[0]}_{testFile.split("/")[-1].split(".")[0]}_{config.dataset}_rationalizedWrong.json',
                    "w",
                ) as fout:
                    json.dump(rationalizedWrongPreds, fout)

    wandb.finish()


if __name__ == "__main__":
    main()
