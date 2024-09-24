# Performance Envelopes of Linear Probes for Latent Representation Edits in GPT Models

## Abstract

The emerging field of representation engineering seeks to promote AI transparency and alignment by developing new methodologies to study the internal representations of large models.  Probing classifiers are a technique for understanding and modifying the operation of neural networks in which a smaller classifier is trained to use the model's internal representation to learn a related probing task.  Similar to a neural electrode array, training probing classifiers can help us both discern and edit the internal representation of a neural network.  This paper presents an evaluation of the use of probing classifiers to modify the internal hidden state of a chess-playing transformer.  The weights of the learned linear classifiers are very informative and can be used to reliably delete pieces from the board showing that the model internally maintains an editable emergent representation of game state.

## Overview

This repository contains the code and probe model tensors for the paper **"Performance Envelopes of Linear Probes for Latent Representation Edits in GPT Models"**. The primary focus is to investigate the application of linear probing classifiers for modifying the internal states of a chess-playing GPT model trained on UCI move sequences. The contributions of this work are as follows:

- A systematic evaluation of the probing classifier's performance across different layers of a transformer model.
- Demonstration of causal effects on the model’s output through targeted modifications to its hidden state.
- Introduction of the legal move probability mass (LMPM) metric to evaluate the semantic validity of outputs post-intervention.

## Repository Structure

- **`/data`**: Currently empty. The unprocessed datasets are available on huggingface.co/austindavis. The processed data is saved here.
- **`/dataset_generation`**: Code to process raw datasets for use by experiments.
- **`/experiments`**: Contains the implementation of the intervention experiments.
- **`/figure-makers`**: Code for making figures used in the paper.
- **`/images`**: Figure makers and experiments output images here.
- **`/modeling`**: Implementation of board state functions, tokenizers, linear probes, and various chess-related utilities.
- **`/models`**: Fully trained probes used during experiemts.
- **`/train_probes`**: Code to train the linear probes.
- **`/train_transformer`**: Code to train the GPT-2 transformer.


## Chess-Playing GPT Model

The GPT model was trained exclusively on tokenized UCI move sequences from 94 million games played on Lichess.org in June 2023. A holdout set of 120,000 games from 2013 was used for evaluation. The model was designed without prior knowledge of chess rules and was trained purely to predict the next token in a sequence.

## Probing Classifiers

Probing classifiers were trained to decode the internal states of the GPT model to classify the piece type and color on each square of the chessboard. The linear classifiers achieved high accuracy and were used to generate intervention vectors that directly modify the model's latent space.

## Key Research Questions

1. **How well does the GPT respond to linear probing interventions at different layers?**
2. **Are the weights of the probe classifier themselves a linear combination of other features in the latent space of the model?**
3. **When the forward pass of the GPT is edited, do the results still obey the domain-specific semantics of chess?**

## Results

- **Classification Performance**: Probing classifiers achieved peak performance between layers 7 and 8 of the model, with an accuracy of over 95% for board state classification. Interestingly, intervention effectiveness peaked earlier.
- **Intervention Effectiveness**: The linear probe-based interventions demonstrated a high success rate, effectively steering the model's behavior towards desired game states while maintaining legal move validity.
- **Linearity of Representations**: The weight matrix of the probe classifiers could be approximated by linear components corresponding to piece type and color, supporting the linear representation hypothesis.

## Conclusion

This work contributes to understanding how latent representations within language models can be both decoded and modified. The findings demonstrate that linear probes are a precise tool for both interpreting and intervening in the model’s internal state, providing a pathway towards more transparent and controllable AI systems.

## Citation

Please cite this paper as:

```
@inproceedings{davisPerformanceEnvelopesLinear2024,
  author={Davis, Austin L and Sukthankar, Gita},
  booktitle={2024 International Conference on Machine Learning and Applications (ICMLA)}, 
  title={Performance Envelopes of Linear Probes for Latent Representation Edits in GPT Models}, 
  year={2024},
  keywords={Representation Engineering; Probing Classifiers; Chess-playing Language Models, GPT},
}
```

## Acknowledgements

Special thanks to the Lichess.org community for providing access to the game data used in this research.