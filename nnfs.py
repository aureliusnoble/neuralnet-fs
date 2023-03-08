import numpy as np

def predict_node (list_x, list_m, b):
  y_pred = 0
  y_pred += b

  for x, m in zip(list_x, list_m):
    y_pred += m * x

  return y_pred

def sigmoid(x):
  sigmoid_x = 1 / (1 + (2.71828**-x))

  return sigmoid_x

def predict_layer(layer):
  input_nodes = layer[0]
  weights = layer[1]
  bias = layer[2]
  num_output_nodes = layer[3]
  apply_transformation = layer[4]

  output_layer = []
  for node in range(num_output_nodes):
    pred_node = predict_node(input_nodes, weights[node], bias)
    if apply_transformation:
      pred_node = sigmoid(pred_node)
    output_layer.append(pred_node)
  
  return output_layer

def feedforward(layers):
  layer_count = len(layers)
  for index, layer in enumerate(layers):
    layer_prediction = predict_layer(layer)

    if index + 1 < layer_count:
      layers[index+1][0] = layer_prediction
    else:
      return layer_prediction[0]

def find_loss(y_pred, y):
  return (y_pred - y)**2

def adjust_bias(layer, layer_index, y, starting_loss):
  bias = layer[2]
  bias_temp = layer[2] + 0.001
  layers_temp = copy.deepcopy(layers)
  layers_temp[layer_index][2] = bias_temp
  new_loss = find_loss(feedforward(layers_temp), y)
  loss_change = new_loss-starting_loss
  bias_updated = bias - loss_change
  return bias_updated

def adjust_weights(layer, layer_index, y, starting_loss):
  layer_weights = layer[1]
  layer_weights_updated = []

  for node_index, node_weight in enumerate(layer_weights):
    node_weights_updated = []

    for weight_index, weight in enumerate(node_weight):
      weight_temp = weight + 0.001
      layers_temp = copy.deepcopy(layers)
      layers_temp[layer_index][1][node_index][weight_index] = weight_temp
      new_loss = find_loss(feedforward(layers_temp), y)
      loss_change = new_loss-starting_loss
      weight_updated = weight - loss_change
      node_weights_updated.append(weight_updated)
      
    layer_weights_updated.append(node_weights_updated)

  return layer_weights_updated

def backpropagation(y, layers):
  starting_loss = find_loss(feedforward(layers), y)
  updated_layers = copy.deepcopy(layers)

  for layer_index, layer in enumerate(layers):
    updated_layers[layer_index][2] = adjust_bias(layer, layer_index, y, starting_loss,)
    updated_layers[layer_index][1] = adjust_weights(layer, layer_index, y, starting_loss)

  return starting_loss, updated_layers


def create_weights(num_output_nodes, input_nodes):
  weights = []
  
  for output_node in range(num_output_nodes):
    node_weights = []

    for input_node in input_nodes:
      random_weight = np.random.rand(1,1)[0][0]
      node_weights.append(random_weight)
    
    weights.append(node_weights)
  
  return weights

def create_network(layers):

  network = []

  for index, layer in enumerate(layers):

    apply_transformation = True

    if (index + 1) > (len(layers) - 1):
      return network

    if (index + 1) == (len(layers) - 1):
      apply_transformation = False
    
    input_nodes = [None] * layer
    num_output_nodes = layers[index+1]
    
    weights = create_weights(num_output_nodes, input_nodes)

    bias = np.random.rand(1,1)[0][0]

    layer = [input_nodes, weights, bias, num_output_nodes, apply_transformation]
    network.append(layer)

def data_loader(inputs, network):
  network[0][0] = inputs
  return network

def unpack_pixels(np_array):
  flat_list = []
  for sub_list in np_array:
    for item in sub_list:
      flat_list.append(item)

  return flat_list

def pack_pixels(input_list, row_len, col_len):
  list_copy = copy.deepcopy(input_list)
  packed_list = []
  for i in range(row_len):
    row_list = []
    for j in range(col_len):
      row_list.append(list_copy.pop(0))
    packed_list.append(row_list)
  
  image = np.array(packed_list)

  return image

def predict_input(input_img, layers):
  layers = data_loader(unpack_pixels(input_img), layers)

  prediction = feedforward(layers)
  return prediction

