Distributed GPU Compute Network
Overview
This project is a distributed compute system that allows users to run machine learning workloads on remote machines across a local network. It enables sharing of idle CPU and GPU resources to execute tasks such as training neural networks.

Features
Remote execution of Python jobs across machines

GPU-aware scheduling using NVML

CLI-based interaction (no UI needed)

Parallel execution across multiple nodes

Dataset splitting for distributed workload simulation

Architecture
Coordinator (Backend)

Accepts jobs

Assigns jobs to nodes

Receives results

Node Agents (Workers)

Poll for jobs

Execute code locally (CPU/GPU)

Send results back

Communication

HTTP over LAN (same WiFi / hotspot)

Tech Stack
Python

FastAPI

Requests

PyTorch

NVML (GPU monitoring)

How It Works
A user submits a job (Python code or training script)

The backend stores the job

Available nodes poll the backend

A node receives the job and executes it

Results (logs/output) are sent back to the backend

GPU Scheduling (NVML)
Each node checks GPU availability before accepting a job using:

GPU utilization

Memory usage

Running processes

A job is only accepted if sufficient GPU resources are available.

Distributed Workload (Demo Logic)
Dataset is split into parts

Each GPU node processes a subset

Example:

Node 1 → first 5000 samples

Node 2 → next 5000 samples

This enables parallel execution and faster completion.

Running the System
1. Start Backend
uvicorn main:app --host 0.0.0.0 --port 8000
2. Run Node Agent (on other machines)
python node_agent.py
Set backend URL inside script:

http://<backend-ip>:8000
3. Submit Job
python submit_job.py
Demo Scenario
Train a model on a single node (baseline)

Split dataset across 2 GPU nodes

Run jobs in parallel

Show reduced execution time

Future Improvements
True distributed training using NCCL

Peer-to-peer architecture (no central server)

Fault tolerance and job rescheduling

Credit-based incentive system

Web dashboard

Key Idea
This system transforms idle machines into a shared compute network, enabling scalable and accessible AI training without relying on centralized cloud infrastructure.
