---
title: "DnD Narrator — Gemini Fine-Tuning MLOps Pipeline"
---

# Overview

This project implements an end-to-end MLOps pipeline for the DnD Narrator AI using Vertex AI Gemini supervised fine-tuning.  
Instead of training models manually with TensorFlow or PyTorch, all model training is executed on Google's LLM backend.  
The pipeline focuses on automating:

- Data ingestion  
- Data preprocessing  
- JSONL dataset generation  
- Triggering Gemini fine-tuning jobs  
- Deploying tuned models to Vertex AI Endpoints  

The goal is to automatically retrain and redeploy the narrator model whenever new story data becomes available.

---

# Setup Environments

In this tutorial, we set up a workflow container to:

- Package Python code for the pipeline  
- Compile the Kubeflow pipeline  
- Submit Gemini fine-tuning jobs  
- Deploy tuned Gemini models  
- Manage authentication and reproducibility  

The workflow container serves as the unified entry point for running the entire MLOps system.

---

# Project Structure

ml-workflow/
├── src/
│ ├── data-collector/
│ ├── data-processor/
│ ├── workflow/
│ └── finetune/
├── secrets/ (local only, not in Git)
└── README.Rmd

# Pipeline Architecture
      Data Collector
             ↓
      Data Processor
  (clean → jsonl → upload)
             ↓
  Gemini Fine-Tune Trigger
             ↓
   Deploy Tuned Model


The pipeline is executed on Vertex AI using Kubeflow Pipelines.

---

# Gemini Fine-Tuning Logic

Training is triggered using:

llm_client <- genai::Client(vertexai = TRUE)
llm_client$tunings$tune(
base_model = "gemini-2.5-flash",
training_dataset = TuningDataset(gcs_uri)
)


All computation and fine-tuning are performed on Google’s backend.

---

# Deployment

After fine-tuning, the tuned Gemini model is automatically deployed to a Vertex AI Endpoint.  
The workflow supports continuous redeployment and model version control.

---

# Run Instructions

## Build individual modules
docker build -t data-collector ./src/data-collector
docker build -t data-processor ./src/data-processor
docker build -t workflow ./src/workflow


## Run workflow container


This container is responsible for compiling and submitting the pipeline.

---

# Key Advantages

- No TensorFlow or PyTorch dependency management  
- No custom training containers needed  
- Training handled entirely by Gemini backend  
- Fully automated retraining pipeline  
- Easy model versioning with Vertex AI  
- Reproducible data and tuning workflow  
- Clean modular architecture suitable for production workflows  

---

# Conclusion

This project demonstrates a modern approach to LLM MLOps:  
automating the data → fine-tune → deploy lifecycle using Vertex AI Gemini.

The pipeline ensures that new data can continuously improve the DnD Narrator model with minimal manual work.

This project demonstrates a modern approach to LLM MLOps:  
automating the data → fine-tune → deploy lifecycle using Vertex AI Gemini.

The pipeline ensures that new data can continuously improve the DnD Narrator model with minimal manual work.