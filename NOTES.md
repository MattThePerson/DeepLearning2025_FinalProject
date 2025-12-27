# Notes

## TODO

- add metrics saving during training (csv)
- add better checkpoint selection
- figure out how to implement attention mechanism (SE or MHA)
<!-- - one good train -->


## MODEL LOADING

Options:
- backbone          `resnet | efficientnet`
- pretrained params `scratch | pretrained_backbone | finetuned`
- param unfreezing  `classifier_only | all`


## MultiHead Attention

> multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)
> attn_output, attn_output_weights = multihead_attn(query, key, value)

