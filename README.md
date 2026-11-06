# Fine-tuning Flan-T5 with LoRA, tracked in MLflow and served with FastAPI

This is my deep learning final project. I fine-tuned a small language model (Flan-T5-small) on two kinds of questions, math word problems and medical questions, logged everything in MLflow, compared the results to the untouched model, and wrote a small API that loads the logged models and answers questions with them.

The notebook I recorded my walkthrough from is here: https://colab.research.google.com/drive/1MihySuLkRr25dXt51vUr9A-xKnWpGZGt?usp=sharing

## What's in the repo

- `dataset_prep.ipynb` cleans the MathQA data and turns it into a Hugging Face dataset.
- `deep_learning_final.ipynb` is the main notebook: LoRA fine-tuning, MLflow logging, and evaluation.
- `app/` is the FastAPI server that serves models from MLflow.
- `workshops/transformer/` is a course notebook where I built GPT-2 from scratch. It's not my own material (see below).

## How it works

**Data.** MathQA is multiple choice, but I wanted the model to write the answer itself. So in `dataset_prep.ipynb` I merged the train and test files (32,822 problems), used a regex to pull out the text of the correct option, and saved a 90/10 train/test split. For the medical side I used `FreedomIntelligence/medical-o1-reasoning-SFT` from the Hugging Face Hub and renamed its columns to match the math data, so one set of helper functions handles both.

**Fine-tuning.** The base model is `google/flan-t5-small` (about 77M parameters). I used LoRA through Hugging Face PEFT with rank 8, alpha 16, and dropout 0.05, applied only to the attention `q` and `v` layers. That leaves 344K trainable parameters out of 77.3M, or 0.45%. Training ran for 5 epochs at a learning rate of 1e-5 using the Hugging Face `Trainer`. To keep Colab runs short I trained on every 5th example, which gave 5,252 training and 657 test examples for math.

**Tracking.** Each run logs its hyperparameters, metrics, the model, and a CSV of the test predictions to MLflow. The MLflow database lives on Google Drive and is exposed through ngrok, so runs don't disappear when the Colab session resets.

**Serving.** The API reads straight from MLflow:

- `GET /models` lists all runs with their params and metrics
- `GET /models/{run_id}` shows details for one run
- `POST /predict` sends your text to one or more models (by run id, or all of them if you leave it empty) and returns each model's answer

If one model fails to load, you get an error message for that model and the others still return results.

## Results

| Task | Metric | Original model | Fine-tuned |
|---|---|---|---|
| Math | exact-match accuracy | 0.91% | 1.37% |
| Medical | ROUGE-1 | 2.54% | about 3% |

These numbers are low, and I don't want to dress that up. Flan-T5-small is tiny, the math problems need several reasoning steps, and exact string match is a harsh way to grade free text. Fine-tuning helped a little on math and barely changed the medical scores. I also tried 1, 5, and 10 epochs and saw no real difference between them.

What I got out of the project was the setup around the model: a pipeline where every experiment is logged, compared against a baseline, and loadable through an API. If I kept going, I'd try a bigger base model, a higher LoRA rank, the full datasets instead of a subsample, and a better metric for math (comparing the final number instead of the whole string).

## Transformer from scratch

`workshops/transformer/transformer_hard.ipynb` comes from the open-source [ML4G course](https://github.com/EffiSciencesResearch/ML4G-2.0). In it I filled in a GPT-2 implementation in PyTorch (embeddings, causal attention, MLP, LayerNorm, unembedding) and checked it against the pretrained GPT-2 weights. The course content belongs to its authors. I included it because it's the architecture that Flan-T5 is built on, and it shows I worked through it by hand.

## Running it

The notebooks are meant for Google Colab with a GPU, Google Drive for storage, and an ngrok token for the MLflow UI. Open the Colab link above and run it from the top.

To run the API on your own machine:

```bash
pip install fastapi uvicorn mlflow transformers pydantic
export MLFLOW_TRACKING_URI=http://localhost:5000   # your MLflow server
uvicorn app.main:app --reload
```

Then open `http://localhost:8000/docs` to try the endpoints. By default the API loads the artifact called `model-flan-t5-small-finetuned`, which is what the notebook logs. You can change it with the `MODEL_ARTIFACT_NAME` environment variable.

## Tools used

Python, PyTorch, Hugging Face (Transformers, Datasets, PEFT, Evaluate), MLflow, FastAPI, Pydantic, pandas, Google Colab, ngrok
