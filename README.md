<div align ="center">
<h1> ControlAR </h1>
<h3> Controllable Image Generation with Autoregressive Models </h3>

Zongming Li<sup>1,\*</sup>, [Tianheng Cheng](https://scholar.google.com/citations?user=PH8rJHYAAAAJ&hl=zh-CN)<sup>1,\*</sup>, Shoufa Chen<sup>2</sup>, Peize Sun<sup>2</sup>, Haocheng Shen<sup>3</sup>,Longjin Ran<sup>3</sup>, Xiaoxin Chen<sup>3</sup>, [Wenyu Liu](http://eic.hust.edu.cn/professor/liuwenyu)<sup>1</sup>, [Xinggang Wang](https://xwcv.github.io/)<sup>1,📧</sup>

<sup>1</sup> Huazhong University of Science and Technology,
<sup>2</sup> The University of Hong Kong
<sup>2</sup> vivo AI Lab

(\* equal contribution, 📧 corresponding author)

[![arxiv paper](https://img.shields.io/badge/arXiv-Paper-red)](https://arxiv.org/abs/2410.02705)
</div>


<div align="center">
<img src="./assets/vis.png">
</div>


## News

`[2024-10-04]:` We have released the [technical report of ControlAR](https://arxiv.org/abs/2410.02705). Code, models, and demos are coming soon!


## Highlights

* ControlAR explores an effective yet simple *conditional decoding* strategy for adding spatial controls to autoregressive models, e.g., [LlamaGen](https://github.com/FoundationVision/LlamaGen), from a sequence perspective.

* ControlAR supports *arbitrary-resolution* image generation with autoregressive models without hand-crafted special tokens or resolution-aware prompts.


## Results

We provide both quantitative and qualitative comparisons with diffusion-based methods in the technical report! 

<div align="center">
<img src="./assets/comparison.png">
</div>


## Acknowledgments

The development of ControlAR is based on [LlamaGen](https://github.com/FoundationVision/LlamaGen), [ControlNet](https://github.com/lllyasviel/ControlNet), [ControlNet++](https://github.com/liming-ai/ControlNet_Plus_Plus), and [AiM](https://github.com/hp-l33/AiM), and we sincerely thank the contributors for thoese great works!

## Citation
If you find ControlAR is useful in your research or applications, please consider giving us a star 🌟 and citing it by the following BibTeX entry.

```bibtex
@article{li2024controlar,
      title={ControlAR: Controllable Image Generation with Autoregressive Models}, 
      author={Zongming Li, Tianheng Cheng, Shoufa Chen, Peize Sun, Haocheng Shen, Longjin Ran, Xiaoxin Chen, Wenyu Liu, Xinggang Wang},
      year={2024},
      eprint={2410.02705},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2410.02705}, 
}
```

