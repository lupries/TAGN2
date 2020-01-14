# Targeted Attentive Graph Neural Network


## Differences to Paper (Zero-Shot Video Object Segmentation using Attentive Graph Neural Networks)

1. message gate (Eq.8): we use pixel-wise gate instead of channel-wise gate (also in github code) [code](https://github.com/carrierlxk/AGNN/blob/master/deeplab/siamese_model_conf_gnn.py#L335)
2. intra-attention (Eq.4): intra-attention is used at every message-passing step (different to code where it is used just once before message passing starts) [code](https://github.com/carrierlxk/AGNN/blob/master/deeplab/siamese_model_conf_gnn.py#L163-L164)
3. message aggregation (Eq.9): we use the average of messages instead of sum (code uses convolutions to aggregate messages) Note: using sum is inconsistent with ConvGRUs and gate [code](https://github.com/carrierlxk/AGNN/blob/master/deeplab/siamese_model_conf_gnn.py#L287-L292)
4. we use dropout before readout function to establish information flow over graph neural network (otherwise main information flow over residual connection)