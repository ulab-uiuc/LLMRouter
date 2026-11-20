# MHLC source and adaptations

The source for the mixed-source generator, labeler, feature encoders, weighted regression head, and two-model benchmark is **Multi-Head Latent Control**. The integration started from commit `a5a1fffc44158d5808d5c8cae4f18c4df675bc09` and the author's working-tree Qwen3.5 trainer, feature extractor, runtime, and recipes. [Source snapshot](https://github.com/Amirhosein-gh98/Multi-Head-Latent-Control/tree/a5a1fffc44158d5808d5c8cae4f18c4df675bc09) · [Paper](https://arxiv.org/abs/2607.14277)

LLMRouter uses vLLM for candidate generation and for frozen-backbone replay during both training and routing. The integration pins the example backbone revision, aligns the extracted token IDs with the processor's mask, saves feature-backend metadata with each head, and exposes the workflow through LLMRouter's CLI. The capability-head architecture and source-dependent labeling rules come from MHLC.

The Qwen3.5-4B thinking-off example selects 40,000 questions. The paper's released heads use the original extraction backend. Source datasets and backbone weights are obtained separately at runtime. [Router guide](../README.md)
