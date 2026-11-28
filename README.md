<!-- PAI-SG-LINKS:BEGIN -->
## Project Links

- [Project Portal](https://statera-workshop.github.io/#repo-mm_fusion_tfn)
- [Statera Workshop Portal](https://statera-workshop.github.io/)
<!-- PAI-SG-LINKS:END -->

# MM_Fusion_TFN

**Tensor Fusion Network on AV-MNIST - convergence study (PAI-SG Multimodal Wave 1)**

Part of the *Physical AI* reference series of the **Statera Working Group**. One method of the multimodal convergence comparison in [MM_Wave1A_AVMNIST](https://github.com/Statera-Workshop/MM_Wave1A_AVMNIST).

## Method

Outer product of [1, image] and [1, audio] (49 x 193 = 9,457-d), MLP 9457-100-10 (MultiBench TensorFusion).

**Common setup** (identical for the AV-MNIST repositories; full comparison in [MM_Wave1A_AVMNIST](https://github.com/Statera-Workshop/MM_Wave1A_AVMNIST)).
AV-MNIST (MultiBench pre-processed release): 28 x 28 PCA-reduced MNIST digit + 112 x 112 spectrogram of a spoken digit
with ESC-50 background noise, 10 classes; train 55,000 / val 5,000 (last 5,000 of the training file) / test 10,000 pairs.
Encoders from scratch: image LeNet(1,6,3) -> 48-d, audio LeNet(1,6,5) -> 192-d (MultiBench); image standardised with
the training mean / std, audio / 255. Adam 1e-3 constant, batch 64, budgets 30 and 100 epochs, seeds 777 / 778 / 779,
fp32, deterministic cuDNN. Every epoch: test accuracy and a frozen linear probe per modality (10,000 training pairs,
standardised features, full-batch Adam, test accuracy). Reported value = last epoch.

**Convergence measures.** Peak test accuracy and its epoch, drop after the peak (peak - final), and the *audio gap* =
audio-only model's probe - this model's audio probe (same budget; positive = audio under-learned in the fusion model).
*BN re-estimated*: the saved last-epoch weights evaluated again after recomputing BatchNorm statistics on 200 training
batches (the series' evaluation practice); for AV-MNIST it changes nothing beyond +/-0.004.

## Results (AV-MNIST, mean of seeds; range in brackets)

| Model | Optimiser | Epochs | Seeds | Final test acc (range) | Peak (epoch) | Drop after peak | Audio probe | Audio gap | Final, BN re-estimated |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Tensor Fusion Network | Adam 1e-3 | 30 | 3 | 0.668 (0.005) | 0.719 (5.7) | 0.051 | 0.380 | +0.007 | 0.669 (0.007) |
| Tensor Fusion Network | Adam 1e-3 | 100 | 3 | 0.670 (0.005) | 0.719 (5.7) | 0.049 | 0.389 | -0.004 | 0.671 (0.006) |

Audio-only reference probe: 0.387 (30 epochs) / 0.385 (100 epochs).

**Finding.** Earliest peak (about epoch 6) and the largest drop after it among the fusion models.

## Weights

Release **v1.0** of this repository (6 files, SHA-256 in `SHA256SUMS.txt`): `avmnist_tfn_adam_e100_s777.pth`, `avmnist_tfn_adam_e100_s778.pth`, `avmnist_tfn_adam_e100_s779.pth`, `avmnist_tfn_adam_e30_s777.pth`, `avmnist_tfn_adam_e30_s778.pth`, `avmnist_tfn_adam_e30_s779.pth`.

```python
import torch, sys; sys.path.insert(0, 'code'); import mm_a as M
m = M.AVNet('<model>')  # img, aud, late, late_lin, tfn, lmf, gblend, ogmge, ogm
m.load_state_dict(torch.load('avmnist_<model>_<opt>_e<epochs>_s<seed>.pth', weights_only=True), strict=True)
```

**Licence note.** The AV-MNIST audio contains ESC-50 noise (CC BY-NC 3.0), so the released weights are for non-commercial use. See THIRD_PARTY.md.

## Run

Data: `gdown 1KvKynJJca5tDtI5Mmp6CoRh9pQywH8Xp -O avmnist.tar.gz` (MultiBench release), unpack to `~/MM_Wave1/data/avmnist`, check with `python code/count_avmnist.py ~/MM_Wave1/data/avmnist`.

```bash
cd code && bash smoke_a.sh   # gates + 1-epoch smoke
python mm_a.py --model tfn --epochs 100 --seed 777
python analyze_a.py; python recal_bn_a.py
```
