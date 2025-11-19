import os
import pickle
import numpy as np

def unpickle(fp):
    with open(fp, 'rb') as f:
        return pickle.load(f, encoding='bytes')

folder = 'data/cifar-10-batches-py'  # path where you extracted the dataset
# Downloadable from: https://www.cs.toronto.edu/~kriz/cifar.html :contentReference[oaicite:1]{index=1}

# Load training batches
X_train_list = []
for i in range(1, 6):
    batch = unpickle(os.path.join(folder, f'data_batch_{i}'))
    X_train_list.append(batch[b'data'])
X_train = np.concatenate(X_train_list, axis=0)  # shape (50000, 3072)

# Load test batch
batch_test = unpickle(os.path.join(folder, 'test_batch'))
X_test = batch_test[b'data']  # shape (10000, 3072)

# Reshape to (N, 32, 32, 3)
X_train = X_train.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
X_test  = X_test.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)

# Ensure dtype uint8
X_train = X_train.astype(np.uint8)
X_test  = X_test.astype(np.uint8)

# Create dictionary
dataset = {
    'train': X_train,
    'test' : X_test
}

# Save
output_path = 'cifar10.pkl'
with open(output_path, 'wb') as f:
    pickle.dump(dataset, f)

print('Saved dataset to', output_path)
print('train shape:', X_train.shape, 'dtype:', X_train.dtype)
print('test  shape:', X_test.shape,  'dtype:', X_test.dtype)
