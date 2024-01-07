import sys
import logging
logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

logging.info("Printing sys.path")
print(sys.path)
logging.info("Importing tensorflow")
import tensorflow as tf
logging.info("Loaded up tensorflow version %s successfully.", tf.__version__)
logging.info("Num GPUs Available: %d", len(tf.config.list_physical_devices('GPU')))

logging.info("Testing pytorch ...")
import torch
logging.info("Loaded pytorch v%s", torch.__version__)
logging.info("Torch can access the GPU: %s", torch.cuda.is_available())

logging.info("Testing torch_scatter ...")
import torch_scatter
logging.info("Loaded torch_scatter v%s", torch_scatter.__version__)
import torch_sparse
logging.info("Loaded torch_sparse v%s", torch_sparse.__version__)
import torch_cluster
logging.info("Loaded torch_cluster v%s", torch_cluster.__version__)
import torch_geometric
logging.info("Loaded torch_geometric v%s", torch_geometric.__version__)

logging.info("Testing JAX on GPU.")
import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random
key = random.PRNGKey(0)
x = random.normal(key, (10,))
print(x)
size = 3000
x = random.normal(key, (size, size), dtype=jnp.float32)
jnp.dot(x, x.T).block_until_ready()  # runs on the GPU{}
logging.info("Loaded JAX v%s", jax.__version__)