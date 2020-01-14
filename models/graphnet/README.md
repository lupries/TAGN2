# Attentive Graph Neural Network in PyTorch

# Install
```
pip install torch-scatter
pip install torch-sparse
pip install torch-cluster
pip install torch-geometric

```
# Usage

```python

from TAGN2.models.graphnet import AGNN

# initialize Graph Network
graph = AGNN(loops=2, channels=5, size=5)
# move model to gpu
graph = graph.cuda()

# create sample data
data = torch.ones(3,5,10,10).cuda()

# run model
output = graph(data)

```
