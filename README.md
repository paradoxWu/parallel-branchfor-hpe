# Parallel-Branch Network for 3D Human Pose and Shape Estimation in Video

![comparsion](images/visualization.png)

## paper to vise

* [X] ST/STB 写法错误
* [X] 解释为什么VIBE的model只选择了第一个
* [X] 还应该解释它的局限性以及本文的方法如何优于郑的工作
* [ ] Ziniu Wan/Lin的论文引用
* [X] 对于损失项，将损失 L_(3d) 和 L_(3d_2) 命名为 L_(3d_STB) 和 L_(3d_SMPL) 似乎更直观。
* [X] 解释为何ST branch 的效果很差这与前面的陈述相矛盾（前面的描述不对
* [X] 图一左下角那个箭头改一下
* [X] 如何计算损失函数 L_3d、L_2d、L_3d_2 和 L_smpl？
* [X] 如何初始化和训练模型，即任何分支都使用预训练的权重？
* [X] 在第 3.2 节中，论文提到了 SMPL 分支中的“简单变压器编码器”。报纸上说它有 4 个头，但除此之外别无他法。我想知道更多关于它的细节
* [X] 第四节讨论模型局限性和改进
* [X] 解释不用h36m的原因
* [ ] 增加mpi-inf-3d关于pck的指标 (不一定需要)
* [ ] 更多的可视化( 准备一个视频可视化出来即可，有时间可以补充上一些图)

## Features

This implementation:

- has the demo and training code for our model implemented purely in PyTorch,
- can work on arbitrary videos with multiple people,
- achieves excellent results on 3DPW and MPI-INF-3DHP datasets,
- includes Temporal SMPLify implementation.
- includes the training code and detailed instruction on how to train it from scratch.
- can create an FBX/glTF output to be used with major graphics softwares.

## Getting Started

Model has been implemented and tested on Ubuntu 18.04 with python >= 3.7. You need a Nvidia GPU.

Clone the repo:

```bash
git clone https://github.com/paradoxWu/parallel-branchfor-hpe.git
```

Install the requirements using `virtualenv` or `conda`:

```bash
# pip
source scripts/install_pip.sh

# conda
source scripts/install_conda.sh
```

## Training

Run the commands below to start training:

```shell
python train.py --cfg configs/config.yaml
```

Note that the training datasets should be downloaded and prepared before running data processing script.
Please see [`doc/train.md`](doc/train.md) for details on how to prepare them.

## Evaluation

Here we compare VIBE with recent state-of-the-art methods on 3D pose estimation datasets. Evaluation metric is
Procrustes Aligned Mean Per Joint Position Error (MPJPE) in mm.

| Models    |  3DPW&#8595;  | MPI-INF-3DHP&#8595; | Human3.6m&#8595; |
| --------- | :------------: | :-----------------: | ---------------- |
| SPIN      |      96.9      |        105.2        |                  |
| Pose2Mesh |      89.2      |          -          |                  |
| VIBE      |      93.5      |        96.6        |                  |
| Ours      | **85.7** |   **95.8**   |                  |

| Models    |  3DPW&#8595;  | MPI-INF-3DHP&#8595; | Human3.6m&#8595; |
| --------- | :------------: | :-----------------: | ---------------- |
| SPIN      |      59.2      |        67.5        | 41.1             |
| Pose2Mesh |      58.3      |          -          | 46.3             |
| VIBE      |      56.5      |   **63.4**   | 41.5             |
| Ours      | **53.1** |         65         |                  |

See [`doc/eval.md`](doc/eval.md) to reproduce the results in this table or
evaluate a pretrained model.

## Models

![network](images/network1.png)

<!-- ## Citation

```bibtex

``` -->

## License

This code is available for **non-commercial scientific research purposes** as defined in the [LICENSE file](LICENSE). By downloading and using this code you agree to the terms in the [LICENSE](LICENSE). Third-party datasets and software are subject to their respective licenses.

## References

We indicate if a function or script is borrowed externally inside each file. Here are some great resources we
benefit:

- Pretrained HMR and some functions are borrowed from [SPIN](https://github.com/nkolot/SPIN).
- SMPL models and layer is from [SMPL-X model](https://github.com/vchoutas/smplx).
- Some functions are borrowed from [Temporal HMR](https://github.com/akanazawa/human_dynamics).
- Some functions are borrowed from [HMR-pytorch](https://github.com/MandyMo/pytorch_HMR).
- Some functions are borrowed from [Kornia](https://github.com/kornia/kornia).
- Pose tracker is from [STAF](https://github.com/soulslicer/openpose/tree/staf).
- Spatial and Temporal transformer modules are set as
- Most code are borrowed from [VIBE](https://github.com/mkocabas/VIBE)
