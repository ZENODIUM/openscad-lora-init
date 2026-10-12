# openscad-lora-init

A 0.6B model can learn a small OpenSCAD grammar, and the LoRA initialization changes how fast that happens. It does not let the model build a part family that was left out of training.

The model is [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) (Apache 2.0). Each example is a plain-English description of a spacer, a plate, or a flange. The model writes a short OpenSCAD program. A prediction counts only if OpenSCAD compiles it, the bounding box matches within 1 mm, and the volume error is smaller than one missing hole.

Both trained runs use the same rank-8 adapter on NF4-dequantized weights. Random init leaves the adapter at zero. Quantization-error init runs four LoftQ iterations so the frozen base plus the adapter starts closer to the original weights. Data is synthetic, and the test parts are disjoint from training. Runs used a free Kaggle T4.

## Full grammar

Train on spacers, plates, and flanges (1000 examples, 180 steps). Test on 36 held-out parts of the same three families.

| Run | Compiled | Geometry match |
|---|---|---|
| Untouched model | 0/36 | 0/36 |
| Random LoRA | 36/36 | 36/36 |
| Quantization-error LoRA | 36/36 | 36/36 |

NF4 leaves a relative weight error of 0.093. Four LoftQ iterations bring that to 0.089. Loss before training is 1.089 versus 1.046, and at step 20 it is 0.029 versus 0.012. By step 180 both losses are about 0, and the test scores tie. On this grammar, 180 steps is enough for either init.

## Held-out flanges

Train on spacers and plates only (600 examples). Flanges use a bolt circle, which never appears in training. Twelve flanges and eight held-out spacers and plates are compiled at four checkpoints.

| Step | Random, trained shapes | Random, flanges | Quant-error, trained shapes | Quant-error, flanges |
|---|---|---|---|---|
| 10 | 0.25 | 0.00 | 0.25 | 0.00 |
| 20 | 0.75 | 0.00 | 1.00 | 0.00 |
| 40 | 1.00 | 0.00 | 1.00 | 0.00 |
| 80 | 1.00 | 0.00 | 1.00 | 0.00 |

The quantization-error adapter fits the seen shapes at step 20. The random adapter fits them at step 40. Failed flanges mix plate-style holes into the wrong solid. Initialization sped up in-distribution learning. It did not supply a construction the data never contained.

## Setup

- Base model: Qwen3-0.6B, fp16 reference and NF4 for the trained runs
- LoRA rank 8, alpha 8, dropout 0, applied to attention and MLP projections
- LoftQ: 4 iterations, rank 8
- Optimizer: AdamW, learning rate 2e-4, batch 1, gradient accumulation 16
- Checker: OpenSCAD 2021.01, mesh volume and bounding box
