import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import nni
import logging

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)


def read_triples_from_file(filename):
    """
    Reads triples from a file.
    Assumes each line of the file is in the format: head relation tail.
    """
    triples = []
    with open(filename, 'r') as f:
        for line in f:
            h, r, t = line.strip().split()
            triples.append((h, r, t))
    return triples


path = 'C:\\Users\\vikrant.singh\\PycharmProjects\\KGFSNAS\\Countries-S1\\'
train_triples_string = read_triples_from_file(path + 'train.txt')
test_triples_string = read_triples_from_file(path + 'test.txt')
valid_triples_string = read_triples_from_file(path + 'valid.txt')

# Combine all triples to extract unique entities and relations
all_triples = train_triples_string + test_triples_string + valid_triples_string

# Get unique entities and relations
all_entities = list(set([h for h, _, _ in all_triples] + [t for _, _, t in all_triples]))
all_relations = list(set([r for _, r, _ in all_triples]))

# Map entities and relations to indices
entity_to_index = {entity: idx for idx, entity in enumerate(all_entities)}
relation_to_index = {relation: idx for idx, relation in enumerate(all_relations)}

# Convert the string triples to their respective indices
train_triples = [(entity_to_index[h], relation_to_index[r], entity_to_index[t]) for h, r, t in train_triples_string]
test_triples = [(entity_to_index[h], relation_to_index[r], entity_to_index[t]) for h, r, t in test_triples_string]
valid_triples = [(entity_to_index[h], relation_to_index[r], entity_to_index[t]) for h, r, t in valid_triples_string]

# Determine the number of unique entities and relations based on the loaded data
num_entities = len(all_entities)
num_relations = len(all_relations)
print(num_entities, num_relations)
print("Number of total triples: ", len(all_triples))
print("Number of training triples: ", len(train_triples))
print("Number of validation triples: ", len(valid_triples))
print("Number of test triples: ", len(test_triples))

embedding_dim = 50
# x_samples = torch.linspace(-1, 1, 50).to(device)

# Create embedding layers

entity_embeddings = nn.Embedding(num_entities, embedding_dim).to(device)
relation_embeddings = nn.Embedding(num_relations, embedding_dim).to(device)

all_entity_indices = list(range(num_entities))


def generate_negative_triples(true_triple, all_entity_indices, num_samples, true_triples_list):
    h, r, t = true_triple
    negative_triples = set()

    while len(negative_triples) < num_samples:  # Generate num_samples number of negative triples
        t_corrupted = random.choice(all_entity_indices)  # Corrupt tail

        if (h, r,
            t_corrupted) not in true_triples_list and t_corrupted != t:
            negative_triples.add((h, r, t_corrupted))

    return list(negative_triples)


def complex_num(embedding, x_samples):
    """
    Computes a complex number representation given an embedding and a set of x_samples.
    Returns a complex number representation in the form of c = cos(x) + i*sin(x).
    """
    # Compute the real and imaginary parts
    real_part = (embedding * torch.cos(x_samples.unsqueeze(-1))).sum(-1)
    imag_part = (embedding * torch.sin(x_samples.unsqueeze(-1))).sum(-1)

    # Combine real and imaginary parts
    # complex_representation = torch.stack((real_part, imag_part), dim=-1)  # Shape: [x_samples, 2]
    # Aggregate real and imaginary parts
    complex_representation = real_part + imag_part  # Shape: [x_samples]

    return complex_representation


# def polynomial(embedding , x_samples):
#     """
#     Computes a polynomial given an embedding and a set of x_samples.
#     Returns a polynomial vector representation.
#     """
#     # The polynomial is in the form of a_0 + a_1*x + a_2*x^2 + ...
#     # print(embedding , "embedding")
#     # Extend dimensions for broadcasting
#     emb_expanded = embedding.unsqueeze(0)  # Shape: [1, embedding_dim]
#     x_expanded = x_samples.unsqueeze(1)  # Shape: [num_samples, 1]
#
#     # Calculate polynomial values using broadcasting
#     powers_of_x = x_expanded ** torch.arange(len(embedding)).to(embedding.device)
#     # Shape: [num_samples, embedding_dim]
#     poly_vector = (emb_expanded * powers_of_x).sum(dim=-1)  # Element-wise multiplication followed by sum
#
#     return poly_vector

def polynomial(embedding, x_samples):

    emb_expanded = embedding.unsqueeze(0)  # Shape: [1, embedding_dim]
    x_expanded = x_samples.unsqueeze(1)  # Shape: [num_samples, 1]

    powers_of_x = x_expanded ** torch.arange(len(embedding)).to(embedding.device)
    poly_vector = (emb_expanded * powers_of_x).sum(dim=-1)

    # Normalize the output vector
    poly_vector = poly_vector / torch.norm(poly_vector)

    return poly_vector



#
# class Mish(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#
#     def forward(self, x):
#         return x * torch.tanh(F.softplus(x))
#
# class NeuralNet(nn.Module):
#
#   def __init__(self, input_dim, output_dim):
#
#     super().__init__()
#
#     # Input dim should match input data
#     l1_dim = input_dim
#
#     # Calculate hidden layer dims
#     # l2_dim = l1_dim // 2
#     # l3_dim = l2_dim // 2
#     l2_dim = l1_dim
#     l3_dim = l2_dim
#
#     # Output dim
#     l4_dim = output_dim
#
#     # Layer definitions
#     self.fc1 = nn.Linear(l1_dim, l2_dim)
#     self.fc2 = nn.Linear(l2_dim, l3_dim)
#     self.fc3 = nn.Linear(l3_dim, l4_dim)
#     self.fc4 = nn.Linear(l4_dim, output_dim)
#
#   def forward(self, input_embedding):
#     x = x_samples
#     combined_input = torch.cat((input_embedding.unsqueeze(0), x.unsqueeze(0)), dim=1)  # Shape: [1, 100]
#     out = Mish()(self.fc1(combined_input))
#     out = Mish()(self.fc2(out))
#     out = Mish()(self.fc3(out))
#     out = self.fc4(out)
#     return out.squeeze(0)  # Shape: [50]

# def forward(self, input_embedding, x_samples):
#
#
#   # Layer 1
#   out1 = torch.tanh(self.fc1(x))
#
#   # Layer 2
#   out2 = torch.tanh(self.fc2(out1))
#
#   # Layer 3
#   out3 = torch.tanh(self.fc3(out2))
#
#   # Layer 4
#   out4 = self.fc4(out3)
#
#   return out4
#
# neural_network = NeuralNet(100, 50)
# neural_network = neural_network.to(device)

# class NeuralNet(nn.Module):
#     def __init__(self, num_layers, layer_size, dropout_rate, activation_function, input_dim):
#         super().__init__()
#
#         self.layers = nn.ModuleList()
#         self.dropout = nn.Dropout(dropout_rate)
#         self.activation_function = activation_function
#
#         # Start with double the input_dim because we concatenate input_embedding and x_samples
#         current_dim = 2 * input_dim
#         for i in range(num_layers - 1):
#             # Intermediate layers can use layer_size
#             self.layers.append(nn.Linear(current_dim, layer_size))
#             current_dim = layer_size
#
#         # The last layer should output the original input_dim
#         self.layers.append(nn.Linear(current_dim, input_dim))
#
#     def forward(self, input_embedding):
#
#         # Adjust x_samples to match the input_embedding size if needed
#         x_samples = x_samples.view(1, -1)  # Ensure it's a row vector for concatenation
#
#         # Concatenate input_embedding with x_samples
#         x = torch.cat((input_embedding.unsqueeze(0), x_samples), dim=1)
#         for layer in self.layers[:-1]:
#             x = layer(x)
#             if self.activation_function == 'relu':
#                 x = torch.relu(x)
#             elif self.activation_function == 'tanh':
#                 x = torch.tanh(x)
#             elif self.activation_function == 'sigmoid':
#                 x = torch.sigmoid(x)
#             x = self.dropout(x)
#
#         # No activation or dropout after the last layer
#         x = self.layers[-1](x)
#         return x.squeeze(0)  # Remove batch dimension


def compute_compostional_score(h_idx, r_idx, t_idx):
    x_samples = torch.linspace(0, 1, 50).to(device)

    h = entity_embeddings(torch.tensor([h_idx]).to(device))[0]
    r = relation_embeddings(torch.tensor([r_idx]).to(device))[0]
    t = entity_embeddings(torch.tensor([t_idx]).to(device))[0]

    fh = FUNCTION_MAP[FUNCTION_SPACE](h, x_samples)
    ft = FUNCTION_MAP[FUNCTION_SPACE](t, x_samples)
    frh = FUNCTION_MAP[FUNCTION_SPACE](r, fh)

    score = torch.trapz(frh * ft, x_samples, dim=0)

    return score


def compute_vtp_score(h_idx, r_idx, t_idx):
    x_samples = torch.linspace(0, 2, 50).to(device)
    # Fetching the embeddings for the head, relation, and tail entities on GPU
    h = entity_embeddings(torch.tensor([h_idx]).to(device))[0]
    r = relation_embeddings(torch.tensor([r_idx]).to(device))[0]
    t = entity_embeddings(torch.tensor([t_idx]).to(device))[0]

    # Transform the embeddings using the polynomial function
    fh = FUNCTION_MAP[FUNCTION_SPACE](h , x_samples)
    fr = FUNCTION_MAP[FUNCTION_SPACE](r , x_samples)
    ft = FUNCTION_MAP[FUNCTION_SPACE](t , x_samples)

    # Compute the VTP score using the transformed embeddings

    # score = torch.sum(fh * (fr * ft))
    score = - torch.trapz(ft, x_samples, dim=0) * torch.trapz(fh * fr, x_samples, dim=0) + torch.trapz(fr, x_samples,
                                                                                                       dim=0) * torch.trapz(
        ft * fh, x_samples, dim=0)
    return score


"""Vector Triple Product (VTP) Scoring Function:

score=∫fh⋅(fr⊙ft)dx
"""


def compute_trilinear_score(h_idx, r_idx, t_idx):
    x_samples = torch.linspace(0, 2, 50).to(device)
    h = entity_embeddings(torch.tensor([h_idx]).to(device))[0]
    r = relation_embeddings(torch.tensor([r_idx]).to(device))[0]
    t = entity_embeddings(torch.tensor([t_idx]).to(device))[0]

    fh = FUNCTION_MAP[FUNCTION_SPACE](h , x_samples)
    fr = FUNCTION_MAP[FUNCTION_SPACE](r , x_samples)
    ft = FUNCTION_MAP[FUNCTION_SPACE](t , x_samples)

    # score = torch.sum(fh * fr * ft)  # Element-wise multiplication across the three vectors
    score = torch.trapz(fh * fr * ft, x_samples, dim=0)

    return score


"""Trilinear Scoring Function:

score=∫fh⋅fr⋅ftdx
"""

# Print weights before training
# print("Weights before training:")
# for name, param in neural_network.named_parameters():
#     print(name, param.data)

# Optimizer

margin = 1.0
criterion = nn.MarginRankingLoss(margin=margin)


def compute_margin_loss(positive_score, negative_score):
    y = torch.ones_like(positive_score)
    loss = criterion(positive_score, negative_score, y)
    return loss


def calculate_loss_BCELogistLoss(pos_scores, neg_scores):
    # Convert scores to probabilities
    pos_probs = torch.sigmoid(pos_scores)
    neg_probs = torch.sigmoid(neg_scores)

    # True labels
    pos_labels = torch.ones_like(pos_scores)
    neg_labels = torch.zeros_like(neg_scores)

    # BCELoss for positive and negative triples
    pos_loss = F.binary_cross_entropy(pos_probs, pos_labels)
    neg_loss = F.binary_cross_entropy(neg_probs, neg_labels)

    # Combine the losses
    total_loss = pos_loss + neg_loss

    return total_loss


def l2_loss(pos_scores, neg_scores):
    margin = 1
    return torch.sum(F.relu(neg_scores - pos_scores + margin))


# Function spaces
POLYNOMIAL = 'polynomial'
COMPLEX = 'complex'
NN = 'neural_network'

# Scoring functions
VTP = 'vtp'
TRILINEAR = 'trilinear'
COMPOSITIONAL = 'compute_score'

# Loss Functions
BCE = "Binary cross entropy with logist loss"
L2 = "L2 Loss"
MARGIN_LOSS_FN = "Margin based loss function"

# Global variables (initial values can be None or some default value)
FUNCTION_SPACE = None
LOSS_FN = None
SCORING_FN = None
FUNCTION_MAP = None
SCORING_MAP = None
LOSS_MAP = None


def configuration(function_space, loss_function, scoring_function):
    global FUNCTION_SPACE, LOSS_FN, SCORING_FN, FUNCTION_MAP, SCORING_MAP, LOSS_MAP

    # Select which to use
    FUNCTION_SPACE = function_space
    LOSS_FN = loss_function
    SCORING_FN = scoring_function

    # Mapping function space name to function
    FUNCTION_MAP = {
        POLYNOMIAL: polynomial,
        COMPLEX: complex_num
        # NN: neural_network
    }

    # Mapping scoring fn name to function
    SCORING_MAP = {
        VTP: compute_vtp_score,
        TRILINEAR: compute_trilinear_score,
        COMPOSITIONAL: compute_compostional_score
    }

    # Mapping loss fn name to function
    LOSS_MAP = {
        BCE: calculate_loss_BCELogistLoss,
        L2: l2_loss,
        MARGIN_LOSS_FN: compute_margin_loss
    }


def compute_MRR(test_triples):
    rr_sum = 0.0

    for h_idx, r_idx, t_true_idx in test_triples:
        scores = []

        # Score all entities as potential tails
        for t_idx in all_entity_indices:
            score = SCORING_MAP[SCORING_FN](h_idx, r_idx, t_idx)
            scores.append((t_idx, score.item()))

        # Sort entities based on their scores
        ranked_entities = sorted(scores, key=lambda x: x[1], reverse=True)
        rank = [idx for idx, (entity, _) in enumerate(ranked_entities) if entity == t_true_idx][0] + 1
        # Add the reciprocal rank to the sum
        rr_sum += 1.0 / rank

    # Compute the mean reciprocal rank
    mrr = rr_sum / len(test_triples)
    return mrr


def forward(triples):
    total_loss = 0.0

    for h_idx, r_idx, t_idx in triples:
        h_idx, r_idx, t_idx = torch.tensor(h_idx).to(device), torch.tensor(r_idx).to(device), torch.tensor(t_idx).to(
            device)
        # Compute positive score
        positive_score = SCORING_MAP[SCORING_FN](h_idx, r_idx, t_idx)
        # print(positive_score , "positive_score" , h_idx , r_idx , t_idx , "ids")

        accumulated_loss = 0.0
        num_neg_samples = 5
        negative_triples = generate_negative_triples((h_idx, r_idx, t_idx), all_entity_indices, num_neg_samples,
                                                     train_triples)

        for h_neg_idx, r_neg_idx, t_neg_idx in negative_triples:
            h_neg_idx, r_neg_idx, t_neg_idx = torch.tensor(h_neg_idx).to(device), torch.tensor(r_neg_idx).to(
                device), torch.tensor(t_neg_idx).to(device)
            # Compute negative score
            negative_score = SCORING_MAP[SCORING_FN](h_neg_idx, r_neg_idx, t_neg_idx)
            # print(negative_score , "negative_score" , h_neg_idx , r_neg_idx , t_neg_idx , "ids")

            # Compute loss for the positive and negative score pair
            loss = LOSS_MAP[LOSS_FN](positive_score, negative_score)
            # print(loss , "loss")
            accumulated_loss += loss

        accumulated_loss /= num_neg_samples
        accumulated_loss.backward()
        # for name, param in neural_network.named_parameters():
        #   if param.grad is not None:
        #      print(name, param.grad)
        optimizer.step()
        optimizer.zero_grad()
        # for param in neural_network.parameters():
        #   param.grad.zero_

        total_loss += accumulated_loss.item()

    return total_loss


# # Training Loop
# num_epochs = 100
#
# for epoch in range(num_epochs):
#     configuration(POLYNOMIAL, BCE, VTP)
#     total_loss = forward(train_triples)
#     print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss}")
#     # print("Weights after epoch {}:".format(epoch+1))
#     # for name, param in neural_network.named_parameters():
#     #   print(name, param.data)
#
# # After training, evaluate MRR on test_triples
# mrr_value = compute_MRR(test_triples)
# print(f"Mean Reciprocal Rank (MRR) on Test Data: {mrr_value}")

configurations = [
    # (POLYNOMIAL, BCE, VTP),
    # (POLYNOMIAL, BCE, TRILINEAR),
    # (POLYNOMIAL, BCE, COMPOSITIONAL),
    # (POLYNOMIAL, L2, VTP),
    # (POLYNOMIAL, L2, TRILINEAR),
    (POLYNOMIAL, L2, COMPOSITIONAL),
    # (POLYNOMIAL, MARGIN_LOSS_FN, VTP),
    # (POLYNOMIAL, MARGIN_LOSS_FN, TRILINEAR),
    (POLYNOMIAL, MARGIN_LOSS_FN, COMPOSITIONAL),
    # (COMPLEX, BCE, VTP),
    # (COMPLEX, BCE, TRILINEAR),
    # (COMPLEX, BCE, COMPOSITIONAL),
    # (COMPLEX, L2, VTP),
    # (COMPLEX, L2, TRILINEAR),
    (COMPLEX, L2, COMPOSITIONAL),
    # (COMPLEX, MARGIN_LOSS_FN, VTP),
    # (COMPLEX, MARGIN_LOSS_FN, TRILINEAR),
    (COMPLEX, MARGIN_LOSS_FN, COMPOSITIONAL)
    # (NN, BCE, VTP),
    # (NN, BCE, TRILINEAR),
    # (NN, BCE, COMPOSITIONAL),
    # (NN, L2, VTP),
    # (NN, L2, TRILINEAR),
    # (NN, L2, COMPOSITIONAL),
    # (NN, MARGIN_LOSS_FN, VTP),
    # (NN, MARGIN_LOSS_FN, TRILINEAR),
    # (NN, MARGIN_LOSS_FN, COMPOSITIONAL)
]


def initialize_weights(embedding):
    nn.init.uniform_(embedding.weight.data, -0.05, 0.05)


# Assuming you have embedding layers named `entity_embedding` and `relation_embedding`:

results = []

#
# tuned_params = nni.get_next_parameter()
# num_layers = tuned_params['num_layers']
# layer_size = tuned_params['layer_size']
# dropout_rate = tuned_params['dropout_rate']
# activation_function = tuned_params['activation_function']
# learning_rate = tuned_params['learning_rate']
#
# # Create the NeuralNet instance with the tuned hyperparameters
# input_dim = 100  # This should be set to the size of your input embedding
#
#
# neural_network = NeuralNet(num_layers, layer_size, dropout_rate, activation_function, input_dim)


optimizer = optim.Adam([
    {'params': entity_embeddings.parameters()},
    {'params': relation_embeddings.parameters()}
    # {'params': neural_network.parameters()}
], lr=0.001)

for config in configurations:
    initialize_weights(entity_embeddings)
    initialize_weights(relation_embeddings)
    configuration(*config)

    # Training Loop (your existing code)
    for epoch in range(100):
        total_loss = forward(train_triples)
        print(f"Epoch {epoch + 1}/{100}, Loss: {total_loss}")
    # Evaluation
    mrr_value = compute_MRR(test_triples)
    # Log the MRR
    logging.info(f'Mean Reciprocal Rank (MRR): {mrr_value}')

    # Report the final result to NNI
    nni.report_final_result(mrr_value)
    print(f"Configuration: {config},(MRR) on Test Data: {mrr_value}")

    # Log the results
    results.append((config, mrr_value))


