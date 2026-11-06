# Fine-tuning Flan-T5 with LoRA, tracked in MLflow and served with FastAPI

A deep learning project that fine-tunes a small language model (Flan-T5-small) on two kinds of questions, math word problems and medical questions. Experiments are logged in MLflow and compared against the untouched model. A small API loads the logged models and answers questions with them.

Walkthrough notebook (Colab): https://colab.research.google.com/drive/1MihySuLkRr25dXt51vUr9A-xKnWpGZGt?usp=sharing

## Repository contents

- `dataset_prep.ipynb`: cleans the MathQA data and converts it to a Hugging Face dataset.
- `deep_learning_final.ipynb`: main notebook covering LoRA fine-tuning, MLflow logging, and evaluation.
- `app/`: FastAPI server that serves models from MLflow.
- `workshops/transformer/`: course notebook implementing GPT-2 from scratch (third-party material, see below).

## Approach

**Data.** MathQA is multiple choice, but the task here is to generate the answer itself. In `dataset_prep.ipynb`, the train and test files are merged (32,822 problems), a regex extracts the text of the correct option, and the result is saved as a 90/10 train/test split. The medical data is `FreedomIntelligence/medical-o1-reasoning-SFT` from the Hugging Face Hub, with columns renamed to match the math data so the same helper functions work for both.

**Fine-tuning.** The base model is `google/flan-t5-small` (about 77M parameters). LoRA is applied through Hugging Face PEFT with rank 8, alpha 16, and dropout 0.05, on the attention `q` and `v` layers only. This leaves 344K trainable parameters out of 77.3M (0.45%). Training runs for 5 epochs at a learning rate of 1e-5 using the Hugging Face `Trainer`. To keep Colab runs short, training uses every 5th example, giving 5,252 training and 657 test examples for math.

**Tracking.** Each run logs its hyperparameters, metrics, the model, and a CSV of test predictions to MLflow. The MLflow database is stored on Google Drive and exposed through ngrok, so runs persist across Colab session resets.

**Serving.** The API reads directly from MLflow:

- `GET /models` lists all runs with their params and metrics
- `GET /models/{run_id}` returns details for one run
- `POST /predict` sends input text to one or more models (by run id, or all models if none are given) and returns each model's answer

If a model fails to load, its entry contains an error message and the other models still return results.

## Results

| Task | Metric | Original model | Fine-tuned |
|---|---|---|---|
| Math | exact-match accuracy | 0.91% | 1.37% |
| Medical | ROUGE-1 | 2.54% | about 3% |

Absolute scores are low. Flan-T5-small is a very small model, the math problems require multi-step reasoning, and exact string match is a strict way to score free-form text. Fine-tuning gave a small improvement on math and almost no change on the medical task. Runs with 1, 5, and 10 epochs showed no meaningful difference.

The main outcome is the pipeline: experiments are logged, compared against a baseline, and loadable through an API. Possible next steps are a larger base model, a higher LoRA rank, training on the full datasets, and a better math metric (comparing the final number instead of the whole string).

## Transformer from scratch

`workshops/transformer/transformer_hard.ipynb` comes from the open-source [ML4G course](https://github.com/EffiSciencesResearch/ML4G-2.0). It implements GPT-2 in PyTorch (embeddings, causal attention, MLP, LayerNorm, unembedding) and checks the result against the pretrained GPT-2 weights. The course content belongs to its authors. It is included as background on the architecture family that Flan-T5 belongs to.

## Running the project

The notebooks are designed for Google Colab with a GPU, Google Drive for storage, and an ngrok token for the MLflow UI. Open the Colab link above and run from the top.

To run the API locally:

```bash
pip install fastapi uvicorn mlflow transformers pydantic
export MLFLOW_TRACKING_URI=http://localhost:5000   # your MLflow server
uvicorn app.main:app --reload
```

Interactive docs are then available at `http://localhost:8000/docs`. By default the API loads the artifact named `model-flan-t5-small-finetuned`, which is what the notebook logs. Set `MODEL_ARTIFACT_NAME` to change it.

## Tools

Python, PyTorch, Hugging Face (Transformers, Datasets, PEFT, Evaluate), MLflow, FastAPI, Pydantic, pandas, Google Colab, ngrok
