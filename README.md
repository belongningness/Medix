# Medix
Medix-R1 Training and Evaluation
This guide describes how to set up the environment, download the datasets, run a 2B DAPO training job, merge the model checkpoint, and evaluate it on VQA-RAD.
1. Create the Conda Environment
conda create -n medix python=3.12 -y
conda activate medix
2. Install Training Dependencies
cd Medix-R1/training

# flash-attn must be installed separately; it is not built during installation.
pip install flash-attn --no-build-isolation
pip install -e .
3. Configure the API Model
The same API model is used for reward scoring during training and answer judging during evaluation. Edit config.py:
LLM_CONFIG = {
    "api_key": "<your-api-key>",
    "model_name": "<model-name>",
    "base_url": "<api-base-url>",
    "temperature": 0.7,
    "max_tokens": 8192,
}
Keep your API key private. Do not commit real credentials to a public repository.
4. Download the Training Dataset
From the repository root:
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='MBZUAI/medix-rl-data', repo_type='dataset', local_dir='data/medix-rl-data')"
5. Download the Evaluation Dataset
From the eval2 directory:
cd eval2

python -c "from huggingface_hub import snapshot_download; snapshot_download('flaviagiammarino/vqa-rad', repo_type='dataset', local_dir='datasets_local/vqa-rad')"
6. Train with the 2B DAPO Example
From the training directory:
bash examples/medix-r1_2b_dapo.sh
Checkpoints are saved under a path similar to:
training/checkpoints/medix-r1_2b_dapo/global_step_XXX/actor/
Replace XXX with the actual training step.
7. Merge the Model Weights
The training checkpoint uses FSDP shards. Merge them into Hugging Face format before using the model with vLLM.
The example below merges the checkpoint at step 600. Run it from the eval2 directory:
bash merge.sh
The script reads checkpoint shards from:
training/checkpoints/medix-r1_2b_dapo/global_step_600/actor/
It writes the merged safetensors files to:
actor/huggingface/
If you want to merge another training step, update the checkpoint path in merge.sh.
8. Evaluate on VQA-RAD
From the eval2 directory, run:
bash eval.sh
The script evaluates the checkpoint configured for the rad_vqa task in three stages:
1. Answer generation: Starts a vLLM server with the merged checkpoint and generates answers for the 451 VQA-RAD test examples.
2. API judging: Uses the model configured in config.py to judge each answer as correct (1) or incorrect (0).
3. Score aggregation: Calculates the overall accuracy and prints a results table.
Results are saved under:
eval2/results/huggingface/
├── huggingface.jsonl       # Model-generated answers
├── huggingface_eval.jsonl  # Per-example judge scores
└── huggingface_score.txt   # Final accuracy
