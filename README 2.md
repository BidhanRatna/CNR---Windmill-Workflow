# CNR Windmill HPC Demo

This repository is a proof of concept for testing whether Windmill can execute a repository workload from instructions defined in `cnr-job.yaml` rather than relying on a task-specific Windmill script with fixed parameters.

## Files

- `triangle_attack.py` — HPC-ready adversarial training and Triangle Attack workload
- `cnr-job.yaml` — job definition consumed by the Windmill orchestrator
- `requirements.txt` — Python dependencies
- `models/` — pretrained model weights required by the workload

## First test

The initial job is intentionally reduced to one model, 2 adversarial-training epochs, 10 samples, and a 2,000-query attack budget. After the orchestration path works, increase those values in `cnr-job.yaml`.

## Required model

Place the pretrained VGG16 MNIST weights at:

`models/mnist_vgg16.pth`

## Expected output

The job writes results under:

- `results/`
- `defended_models/`
- `checkpoints/`
