import numpy as np




def universal_classif_accuracy(
  pred_label,
  pred_domain,
  query_label,
  query_domain,
):
  #supports query labels that are lists

  if np.any(pred_label == query_label) and pred_domain == query_domain:
    return 1.0
  else:
    return 0.0



def universal_mmp_at_k(
  top_k,
  actual_top_k,
  num_index_label,
  num_true_index_label,
  nearest,
  query_label,
  query_domain,
  throw_first,
):
 
  #mMP@K calc
  num_correct = 0
  relevances = [] #defined for each query

  #supports query labels that are lists
  
  for j in range(min(top_k,num_index_label)):
    #for every neighbor
    if np.any(query_label == nearest[j][0]) and query_domain == nearest[j][1]:
      num_correct += 1
      relevances.append(1)
    else:
      relevances.append(0)

  if throw_first: 
  #throw away the top neighbor, which must be itself
    num_correct -= 1
    relevances = relevances[1:]

  ##mmp@k calc
  if num_true_index_label == 0:
    mp = 0
  else:
    mp = (num_correct * 1.0) / min(num_true_index_label, actual_top_k)

  #TODO: assert here that the metric is >= 0

  return mp,relevances



def universal_mean_prec_at_R(
  top_k,
  num_index_label,
  num_true_index_label,
  nearest,
  query_label,
  query_domain,
  throw_first,
):
 
  #precR calc
  num_correct = 0

  #supports query labels that are lists
  
  #for this, I should be calculating different number of neighbors for each query,
  #or at least, calculate for all queries as many as the maximum number of positives across all queries
  #so definition is not correct for now

  #it is actually correct recall@K now

  #for j in range(num_index_label):
  for j in range(top_k):

    #for every neighbor
    if np.any(query_label == nearest[j][0]) and query_domain == nearest[j][1]:
      num_correct += 1


  if throw_first:
    num_correct -= 1

  ##precR calc
  #precR += (num_correct * 1.0) / num_true_index_label #was only this before

  #temporary fix: (For knn on train set where classes might only have one sample and query set is index set)
  #for these samples, i return 0 score for now
  if num_true_index_label == 0:
    precR = 0
  else:
    precR = (num_correct * 1.0) / num_true_index_label

  return precR




def universal_map_at_k(
  top_k,
  actual_top_k,
  num_true_index_label,
  nearest,
  query_label,
  query_domain,
  throw_first,
):
  
  relevances_map = [] #defined for each query

  #supports query labels that are lists

  #for loop on the neighbors of that query
  for j in range(top_k):
    
    #for every neighbor
    if np.any(query_label == nearest[j][0]) and query_domain == nearest[j][1]:
      relevances_map.append(1)
    else:
      relevances_map.append(0)

  # Remove the offset.
  if throw_first:
    relevances_map = relevances_map[1:]
  
  ##map@k calc
  assert len(relevances_map) == actual_top_k

  prec = np.cumsum(relevances_map) / (1+np.array(np.arange(len(relevances_map))))
  
  if num_true_index_label == 0:
    ap = 0
  else:
    ap = ((prec * relevances_map).sum()) / min(num_true_index_label, actual_top_k)

  return ap,relevances_map



#follows metric learning Recall@k definition
def universal_recall_at_k(
  top_k,
  nearest,
  query_label,
  query_domain,
  throw_first,
):
 
  #R@K calc
  num_correct = 0

  #supports query labels that are lists
  for j in range(top_k):
    #for every neighbor
    if np.any(query_label == nearest[j][0]) and query_domain == nearest[j][1]:
      num_correct += 1

  if throw_first:
    num_correct -= 1

  rk = 1.0*(num_correct>0)

  return rk
