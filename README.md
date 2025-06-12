# LLM-ESR-Pro

This is a collaborative final project for **CSCI-SHU 381: Recommendation Systems** and **DATS-SHU 369: Machine Learning with Graphs**, supervised by **Prof. Hongyi Wen** and **Prof. Qiaoyu Tan** at NYU Shanghai.

We extend the original [LLM-ESR](https://github.com/liuqidong07/LLM-ESR) framework to improve sequential recommendation performance on users and items.

---

## Key Contributions

- **Attribute-aware Graph Embedding**  
  We construct a heterogeneous item-attribute graph (e.g., brand, category, price) and apply a GNN encoder (GAT) with contrastive learning to inject structural semantics into item representations. These embeddings are fused with LLM-based semantic representations.

- **User-type-aware Prediction**  
  To better model heterogeneous user behaviors, we introduce separate prediction heads for head and tail users. A weighted fusion mechanism is applied during inference, improving personalization and long-tail recall.

---

## Takeaways

1. **Attribute-aware Graph Embedding** introduces structural inductive bias by explicitly modeling item-attribute relationships. However, when the attribute information is sparse or noisy, the learned item representations may be degraded overall, resulting in reduced recommendation performance.

2. **Tail-aware Multi-head Prediction** demonstrates consistent and significant improvements. By separating prediction heads for head and tail users, the model better captures user-specific behavior patterns. For head users with rich interaction histories, skipping self-distillation avoids redundancy and preserves personalized signals.

---

## Citation

If you find this project helpful, please consider citing the original work:

```bibtex
@article{liu2024large,
  title={Large Language Models Enhanced Sequential Recommendation for Long-tail User and Item},
  author={Liu, Qidong and Wu, Xian and Zhao, Xiangyu and Wang, Yejing and Zhang, Zijian and Tian, Feng and Zheng, Yefeng},
  journal={arXiv preprint arXiv:2405.20646},
  year={2024}
}
