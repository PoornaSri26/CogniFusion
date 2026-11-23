# PRD - CogniFusion

## Overview
CogniFusion is a multimodal cognitive-load estimation system fusing EEG (Muse headband, 4 channels) and webcam eye-tracking into a single low/medium/high load classifier, with real-time inference, session logging, and a research-grade Colab evaluation pipeline.

## Problem
Single-modality cognitive-load estimates are unreliable: EEG is noisy on dry electrodes; gaze alone misses internal load. Operators (students, drivers, surgeons, analysts) need a trustworthy, explainable, real-time load signal.

## Goals (measurable)
- Subject-wise CV macro-F1 >= 0.90 on synthetic; honest (non-inflated) metrics on real data
- End-to-end Colab run: clone -> Open in Colab -> Run all -> trained ensemble + full report
- Real-time inference <= 50 ms/epoch on CPU
- Every prediction logged with confidence and modality-trust weights

## Users
Researchers (primary), students, developers building attention-aware apps.

## Features (MoSCoW)
- Must: feature extraction (168 dims), subject-wise CV training, ensemble checkpoint, real-time pipeline, SQLite/session logging, Colab notebook
- Should: GRL/ablation/SWA/KD/EMA/feat-aug variants, conformal prediction, SHAP, ONNX export, real Zenodo dataset path
- Could: dashboard Streamlit UI, edge deployment, more modalities (PPG/IMU)
- Won't (v1): hardware-validated claims, mobile app, multi-user server

## Success metrics
- CV accuracy/F1 reported per fold, pooled, and with bootstrap CIs
- Zero subject leakage (verified by subject-wise splits)
- Notebook executes end-to-end without manual fixes

## Out of scope
Clinical diagnosis, real-time neurofeedback control, hardware certification.
