from os import path
import numpy as np
import pmf

# filename = 'data-500-500-3.txt'
filename = 'ml-100k.txt'
rows = np.loadtxt(filename)
header, examples = rows[0], [[int(row[0]), int(row[1]), float(row[2])] for row in rows[1:]]
user_count, item_count, example_count = header.astype(int)
rank = 10
train, test = examples[0:example_count//2], examples[example_count//2:]

model_p = pmf.model_params(user_count, item_count, rank)
user_p = pmf.user_params(user_count, rank)
item_p = pmf.item_params(item_count, rank)
for it in range(10):
  u, v = user_p[0].copy(), item_p[0].copy()
  train_error = pmf.error(train, user_p[0], item_p[0])
  test_error = pmf.error(test, user_p[0], item_p[0])
  
  print ('%.4f %.4f' % (train_error, test_error))
  user_p, item_p = pmf.expectation(train, *model_p, *user_p, *item_p)
  model_p = pmf.maximization(train, *model_p, *user_p, *item_p)
  
  norm = np.linalg.norm
  unorm = np.abs(norm(u, axis=1) - norm(user_p[0], axis=1))
  vnorm = np.abs(norm(v, axis=1) - norm(item_p[0], axis=1))
  
  if np.max(unorm) < 1e-2 and np.max(vnorm) < 1e-2:
    break
  