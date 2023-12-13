from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
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


path = 'C:\\Users\\vikrant.singh\\PycharmProjects\\KGFSNAS\\UMLS\\'
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

dummy_triples = train_triples
true_triples_set = set(map(tuple, dummy_triples))

# Convert triples to tensor
triples_tensor = torch.tensor(dummy_triples)
test_triples_tensor = torch.tensor(test_triples)


# DataLoader
class TriplesDataset(Dataset):
    def __init__(self, triples):
        self.triples = triples

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        return self.triples[idx]


train_dataset = TriplesDataset(triples_tensor)
train_loader = DataLoader(train_dataset, batch_size=32 , drop_last=False)

entity_embeddings = nn.Embedding(num_entities, embedding_dim).to(device)
relation_embeddings = nn.Embedding(num_relations, embedding_dim).to(device)


# def generate_negative_triples_batch(positive_triples_batch, num_entities, true_triples_set, k):
#     negative_triples_batch = []
#
#     if positive_triples_batch.is_cuda:
#         positive_triples_batch = positive_triples_batch.cpu()
#
#     for h, r, t in positive_triples_batch.numpy():
#         negatives_for_this_triple = 0
#         while negatives_for_this_triple < k:
#             negative_tail = torch.randint(0, num_entities, (1,)).item()
#             if negative_tail != t and (h, r, negative_tail) not in true_triples_set:
#                 negative_triples_batch.append((h, r, negative_tail))
#                 negatives_for_this_triple += 1
#
#     # Convert to tensor and move to the same device as positive_triples_batch
#     return torch.tensor(negative_triples_batch).to(positive_triples_batch.device)

def generate_negative_triples_batch(positive_triples_batch, num_entities, true_triples_set, k):
    negative_triples_batch = []

    if positive_triples_batch.is_cuda:
        positive_triples_batch = positive_triples_batch.cpu()

    for h, r, t in positive_triples_batch.numpy():
        negatives_for_this_triple = 0
        while negatives_for_this_triple < k:
            # Randomly decide to corrupt either the head or the tail
            corrupt_head = torch.rand(1).item() < 0.5

            if corrupt_head:
                negative_head = torch.randint(0, num_entities, (1,)).item()
                if negative_head != h and (negative_head, r, t) not in true_triples_set:
                    negative_triples_batch.append((negative_head, r, t))
                    negatives_for_this_triple += 1
            else:
                negative_tail = torch.randint(0, num_entities, (1,)).item()
                if negative_tail != t and (h, r, negative_tail) not in true_triples_set:
                    negative_triples_batch.append((h, r, negative_tail))
                    negatives_for_this_triple += 1

    return torch.tensor(negative_triples_batch).to(positive_triples_batch.device)


def complex_num_functionSpace_batch(embedding, x_samples_batch):
    real_part = embedding * torch.cos(x_samples_batch)
    imag_part = embedding * torch.sin(x_samples_batch)
    complex_representation = real_part + imag_part

    return complex_representation


def polynomial_functionSpace_batch(embedding, x_samples):
    powers_of_x = torch.stack([x_samples ** i for i in range(embedding.size(1))], dim=-1)
    emb_expanded = embedding.unsqueeze(1)
    poly_vector = emb_expanded * powers_of_x
    poly_vector = poly_vector.sum(dim=-1)
    poly_vector = F.normalize(poly_vector, p=2, dim=-1)

    return poly_vector


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
#             self.layers.append(nn.Linear(current_dim, layer_size))
#             current_dim = layer_size
#
#         self.layers.append(nn.Linear(current_dim, input_dim))
#
#     def forward(self, input_embedding, x_samples):
#         # Concatenate input_embedding with x_samples along the last dimension
#         x = torch.cat((input_embedding, x_samples), dim=-1)
#
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
#         x = self.layers[-1](x)
#         return x

# below cass shows differnt input layers for embedding and x_samples

class NeuralNet(nn.Module):
    def __init__(self, num_layers, layer_size, dropout_rate, activation_function, input_dim):
        super(NeuralNet, self).__init__()

        self.embedding_layer = nn.Linear(input_dim, layer_size)
        self.x_layer = nn.Linear(50, layer_size)
        # write the comment what next line does
        self.combined_layers = nn.ModuleList()

        # Additional layers after combining
        current_dim = 2 * layer_size
        # current_dim = 3 * layer_size
        for i in range(num_layers - 1):
            self.combined_layers.append(nn.Linear(current_dim, layer_size))
            current_dim = layer_size

        self.final_layer = nn.Linear(current_dim, 50)
        self.dropout = nn.Dropout(dropout_rate)
        self.activation = getattr(nn.functional, activation_function)

    def forward(self, input_embedding, x_samples):

        if x_samples.dim() > 1:
            x_samples = x_samples.view(input_embedding.shape[0], -1)

        # Process embedding and x separately
        emb_output = self.embedding_layer(input_embedding)
        x_output = self.x_layer(x_samples)

        # Combine the outputs
        combined = torch.cat([emb_output, x_output], dim=1)

        # Pass through combined layers
        for layer in self.combined_layers:
            combined = self.dropout(self.activation(layer(combined)))

        # Final output
        output = self.final_layer(combined)
        return output
# import numpy as np
# class FunctionSpaceNN(nn.Module):
#     def __init__(self, num_layers, layer_size, dropout_rate, activation_function, embedding_dim, x_sample_features):
#         super(FunctionSpaceNN, self).__init__()
#         self.num_layers = num_layers
#         self.layer_size = layer_size
#         self.dropout_rate = dropout_rate
#         self.activation = getattr(F, activation_function)
#         self.x_sample_features = x_sample_features
#         self.embedding_dim = embedding_dim
#
#
#         # Calculate the size of each weight chunk
#         self.chunk_size = embedding_dim // num_layers
#
#         # x_samples processing layer
#         self.x_samples_layer = nn.Linear(x_sample_features, layer_size)
#
#         # Layers after combining x_samples output and weight chunk
#         self.combined_layers = nn.ModuleList()
#         for _ in range(num_layers):
#             self.combined_layers.append(nn.Linear(self.chunk_size + layer_size, layer_size))
#
#         self.dropout = nn.Dropout(dropout_rate)
#
#     def forward(self, weights, x):
#         k = int(np.sqrt(self.embedding_dim // 2))
#         # Split the input embedding (weights) into chunks
#         n = len(weights)
#         # Weights for two linear layers.
#         w1, w2 = torch.hsplit(weights, 2)
#         # (1) Construct two-layered neural network
#         w1 = w1.view(n, k, k)
#         w2 = w2.view(n, k, k)
#         # (2) Perform the forward pass
#         out1 = torch.tanh(w1 @ x)
#         out2 = w2 @ out1
#         return out2
#
#         # weight_chunks = torch.chunk(input_embedding, self.num_layers, dim=1)
#         #
#         # # Process x_samples
#         # x_output = self.x_samples_layer(x_samples)
#         #
#         # # Iteratively process each weight chunk
#         # for i, layer in enumerate(self.combined_layers):
#         #     combined_input = torch.cat([weight_chunks[i], x_output], dim=1)
#         #     x_output = layer(combined_input)
#         #     if i < self.num_layers - 1:  # Apply activation except for the last layer
#         #         x_output = self.activation(x_output)
#         #     x_output = self.dropout(x_output)
#
#         # return x_output
# below class shows differnt input layers for embedding and x_samples with batch normalization
# class NeuralNet(nn.Module):
#     def __init__(self, num_layers, layer_size, dropout_rate, activation_function, input_dim):
#         super(NeuralNet, self).__init__()
#
#         self.embedding_layer = nn.Linear(input_dim, layer_size)
#         self.x_layer = nn.Linear(50, layer_size)
#         self.combined_layers = nn.ModuleList()
#         self.batch_norm_layers = nn.ModuleList()
#
#         # Additional layers after combining
#         current_dim = 2 * layer_size
#         for i in range(num_layers - 1):
#             self.combined_layers.append(nn.Linear(current_dim, layer_size))
#             self.batch_norm_layers.append(nn.BatchNorm1d(layer_size))
#             current_dim = layer_size
#
#         self.final_layer = nn.Linear(current_dim, 50)
#         self.dropout = nn.Dropout(dropout_rate)
#         self.activation = getattr(nn.functional, activation_function)
#
#     def forward(self, input_embedding, x_samples):
#         if x_samples.dim() > 1:
#             x_samples = x_samples.view(input_embedding.shape[0], -1)
#
#         # Process embedding and x separately
#         emb_output = self.embedding_layer(input_embedding)
#         x_output = self.x_layer(x_samples)
#
#         # Combine the outputs
#         combined = torch.cat([emb_output, x_output], dim=1)
#
#         # Pass through combined layers with Batch Normalization
#         for layer, bn_layer in zip(self.combined_layers, self.batch_norm_layers):
#             combined = layer(combined)
#             combined = bn_layer(combined)
#             combined = self.dropout(self.activation(combined))
#
#         # Final output
#         output = self.final_layer(combined)
#         return output


# below class shows multiplication of embedding and x_samples
# class NeuralNet(nn.Module):
#     def __init__(self, num_layers, layer_size, dropout_rate, activation_function, input_dim):
#         super().__init__()
#
#         self.layers = nn.ModuleList()
#         self.dropout = nn.Dropout(dropout_rate)
#         self.activation_function = activation_function
#
#         # Initialize layers
#         current_dim = input_dim  # Dimension after element-wise multiplication
#         for i in range(num_layers - 1):
#             self.layers.append(nn.Linear(current_dim, layer_size))
#             current_dim = layer_size
#
#         self.layers.append(nn.Linear(current_dim, input_dim))
#
#     def forward(self, input_embedding, x_samples):
#         # Ensure x_samples and input_embedding are of the same shape
#         if x_samples.dim() == 1:
#             x_samples = x_samples.view(-1, 1).expand_as(input_embedding)
#
#         # Element-wise multiplication
#         combined_input = input_embedding * x_samples
#
#         # Pass through the neural network
#         x = combined_input
#         for layer in self.layers[:-1]:
#             x = layer(x)
#             if self.activation_function == 'relu':
#                 x = F.relu(x)
#             elif self.activation_function == 'tanh':
#                 x = F.tanh(x)
#             elif self.activation_function == 'sigmoid':
#                 x = F.sigmoid(x)
#             x = self.dropout(x)
#
#         x = self.layers[-1](x)
#         return x

# below class of nn showds parallel processing of embedding and x_samples

# class NeuralNet(nn.Module):
#     def __init__(self, num_layers, layer_size, dropout_rate, activation_function, input_dim):
#         super().__init__()
#         self.embedding_net = self._build_network(num_layers, layer_size, dropout_rate, activation_function, input_dim)
#         self.x_net = self._build_network(num_layers, layer_size, dropout_rate, activation_function, input_dim)
#         self.final_layer = nn.Linear(2 * layer_size, input_dim)
#         self.dropout = nn.Dropout(dropout_rate)
#         self.activation = getattr(nn.functional, activation_function)
#
#     def _build_network(self, num_layers, layer_size, dropout_rate, activation_function, input_dim):
#         layers = nn.ModuleList()
#         for i in range(num_layers):
#             layers.append(nn.Linear(input_dim if i == 0 else layer_size, layer_size))
#         return layers
#
#     def forward(self, input_embedding, x_samples):
#         # Process input_embedding through embedding_net
#         emb_output = input_embedding
#         for layer in self.embedding_net:
#             emb_output = self.dropout(self.activation(layer(emb_output)))
#
#         # Process x_samples through x_net
#         x_output = x_samples if x_samples.dim() > 1 else x_samples.view(-1, 1)
#         for layer in self.x_net:
#             x_output = self.dropout(self.activation(layer(x_output)))
#
#         # Combine and process through the final layer
#         combined = torch.cat([emb_output, x_output], dim=1)
#         output = self.final_layer(combined)
#         return output

def new_scoring_function(h,r,t):
    x_samples = torch.linspace(0, 1, steps=50).to(h.device)
    x_samples_batch = x_samples.unsqueeze(0).expand(h.size(0), -1)
    # Element-wise multiplication between head and relation

    fh = FUNCTION_MAP[FUNCTION_SPACE](h, x_samples_batch)
    fr = FUNCTION_MAP[FUNCTION_SPACE](r, x_samples_batch)
    ft = FUNCTION_MAP[FUNCTION_SPACE](t, x_samples_batch)
    hr_interaction = fh * fr

    # A transformation on the tail embedding, for example, a non-linear activation
    transformed_tail = torch.tanh(ft)

    # The final score could be the dot product between the transformed tail and the hr_interaction
    score = torch.trapz(hr_interaction * transformed_tail,x_samples, dim=1)

    return score


def compute_compostional_score_batch(h, r, t):
    x_samples = torch.linspace(0, 1, steps=50).to(h.device)
    x_samples_batch = x_samples.unsqueeze(0).expand(h.size(0), -1)

    fh = FUNCTION_MAP[FUNCTION_SPACE](h, x_samples_batch)
    frfh = FUNCTION_MAP[FUNCTION_SPACE](r, fh)
    ft = FUNCTION_MAP[FUNCTION_SPACE](t, x_samples_batch)

    score = torch.trapz(frfh * ft, x_samples, dim=1)

    return score


def compute_vtp_score(h, r, t):
    x_samples = torch.linspace(0, 1, steps=50).to(h.device)
    x_samples_batch = x_samples.unsqueeze(0).expand(h.size(0), -1)

    fh = FUNCTION_MAP[FUNCTION_SPACE](h, x_samples_batch)
    fr = FUNCTION_MAP[FUNCTION_SPACE](r, x_samples_batch)
    ft = FUNCTION_MAP[FUNCTION_SPACE](t, x_samples_batch)

    score = - torch.trapz(ft, x_samples, dim=1) * torch.trapz(fh * fr, x_samples, dim=1) + torch.trapz(fr, x_samples,
                                                                                                       dim=1) * torch.trapz(
        ft * fh, x_samples, dim=1)
    return score


def compute_trilinear_score(h, r, t):
    x_samples = torch.linspace(0, 1, steps=50).to(h.device)
    x_samples_batch = x_samples.unsqueeze(0).expand(h.size(0), -1)

    fh = FUNCTION_MAP[FUNCTION_SPACE](h, x_samples_batch)
    fr = FUNCTION_MAP[FUNCTION_SPACE](r, x_samples_batch)
    ft = FUNCTION_MAP[FUNCTION_SPACE](t, x_samples_batch)

    score = torch.trapz(fh * fr * ft, x_samples, dim=1)

    return score


# Loss functions
def compute_margin_loss_batch(positive_scores, negative_scores, k):
    # Reshape positive_scores to compare against each negative score
    device = positive_scores.device
    negative_scores = negative_scores.to(device)

    positive_scores_repeated = positive_scores.unsqueeze(1).repeat(1, k)

    # Calculate the margin ranking loss for each positive-negative pair
    loss = F.margin_ranking_loss(
        positive_scores_repeated.view(-1),
        negative_scores.view(-1),
        torch.ones(positive_scores_repeated.numel()).to(device),
        margin=1.0,
        reduction='none'
    )

    # Average the loss over the number of negatives per positive
    loss = loss.view(-1, k).mean(dim=1).mean()
    return loss


def calculate_loss_BCELogistLoss_batch(positive_scores, negative_scores, k):
    device = positive_scores.device
    negative_scores = negative_scores.to(device)

    # Reshape positive_scores to compare against each negative score
    positive_scores_repeated = positive_scores.unsqueeze(1).repeat(1, k).view(-1)

    # Convert scores to probabilities
    pos_probs = torch.sigmoid(positive_scores_repeated)
    neg_probs = torch.sigmoid(negative_scores.view(-1))

    # True labels
    pos_labels = torch.ones_like(pos_probs)
    neg_labels = torch.zeros_like(neg_probs)

    # BCELoss for positive and negative triples
    pos_loss = F.binary_cross_entropy(pos_probs, pos_labels, reduction='none')
    neg_loss = F.binary_cross_entropy(neg_probs, neg_labels, reduction='none')

    # Average the losses over the number of negatives per positive
    total_loss = (pos_loss + neg_loss).view(-1, k).mean(dim=1).mean()

    return total_loss


def l2_loss_batch(positive_scores, negative_scores, k):
    device = positive_scores.device
    negative_scores = negative_scores.to(device)

    # Reshape positive_scores to compare against each negative score
    positive_scores_repeated = positive_scores.unsqueeze(1).repeat(1, k).view(-1)

    # Calculate the difference between squares of positive and negative scores
    squared_diffs = positive_scores_repeated ** 2 - negative_scores.view(-1) ** 2

    # Adding a small epsilon for numerical stability
    epsilon = 1e-8
    loss = torch.sqrt(torch.abs(squared_diffs) + epsilon)

    # Average the loss over the number of negatives per positive
    avg_loss = loss.view(-1, k).mean(dim=1).mean()

    return avg_loss


# Function spaces
POLYNOMIAL = 'polynomial'
COMPLEX = 'complex'
NEURALNETWORK = 'neural_network'

# Scoring functions
VTP = 'vtp'
TRILINEAR = 'trilinear'
COMPOSITIONAL = 'compute_score'
NEWSCORE = 'new_score'

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
        POLYNOMIAL: polynomial_functionSpace_batch,
        COMPLEX: complex_num_functionSpace_batch,
        NEURALNETWORK: neural_network
    }

    # Mapping scoring fn name to function
    SCORING_MAP = {
        VTP: compute_vtp_score,
        TRILINEAR: compute_trilinear_score,
        COMPOSITIONAL: compute_compostional_score_batch,
        NEWSCORE: new_scoring_function
    }

    # Mapping loss fn name to function
    LOSS_MAP = {
        BCE: calculate_loss_BCELogistLoss_batch,
        L2: l2_loss_batch,
        MARGIN_LOSS_FN: compute_margin_loss_batch
    }


def forward_batch(triples_batch, optimizer):
    # Ensure the batch is on the correct device
    triples_batch = triples_batch.to(device)

    # Get positive embeddings
    h_embeddings = entity_embeddings(triples_batch[:, 0])
    r_embeddings = relation_embeddings(triples_batch[:, 1])
    t_embeddings = entity_embeddings(triples_batch[:, 2])

    # Compute scores for positive triples
    positive_scores = SCORING_MAP[SCORING_FN](h_embeddings, r_embeddings, t_embeddings)

    # Generate negative triples for the batch, k negative samples per positive sample
    negative_triples_batch = generate_negative_triples_batch(triples_batch, num_entities, true_triples_set, k=5).to(
        device)

    # Get negative embeddings
    h_negative_embeddings = entity_embeddings(negative_triples_batch[:, 0])
    r_negative_embeddings = relation_embeddings(negative_triples_batch[:, 1])
    t_negative_embeddings = entity_embeddings(negative_triples_batch[:, 2])

    # Compute scores for negative triples
    negative_scores = SCORING_MAP[SCORING_FN](h_negative_embeddings, r_negative_embeddings, t_negative_embeddings)

    # Compute batch loss using the custom margin loss function
    batch_loss = LOSS_MAP[LOSS_FN](positive_scores, negative_scores, k=5)

    # Backpropagate and update weights
    # optimizer.zero_grad()
    # batch_loss.backward()
    # optimizer.step()

    return batch_loss

def forward_k_vs_all(triples_batch):
    # Ensure the batch is on the correct device
    triples_batch = triples_batch.to(device)

    # Get positive embeddings
    h_embeddings = entity_embeddings(triples_batch[:, 0])
    r_embeddings = relation_embeddings(triples_batch[:, 1])
    t_embeddings = entity_embeddings(triples_batch[:, 2])

    # Compute scores for positive triples
    positive_scores = SCORING_MAP[SCORING_FN](h_embeddings, r_embeddings, t_embeddings)

    # Initialize tensor for all possible negative scores
    all_entities = torch.arange(num_entities, device=device)
    negative_scores = torch.zeros(triples_batch.size(0), num_entities, device=device)

    # Compute scores for all possible negative triples
    for i in range(num_entities):
        t_negative_embeddings = entity_embeddings(torch.full_like(triples_batch[:, 0], i))
        negative_scores[:, i] = SCORING_MAP[SCORING_FN](h_embeddings, r_embeddings, t_negative_embeddings).squeeze()

    # Calculate loss - here you need to define how you calculate the loss for k vs all approach
    # One approach is to treat it as a multi-class classification problem
    target_labels = triples_batch[:, 2]  # The true tail entity indices
    loss = F.cross_entropy(negative_scores, target_labels)

    return loss

configurations = [
    # (POLYNOMIAL, BCE, VTP),
    # (POLYNOMIAL, BCE, TRILINEAR),
    # (POLYNOMIAL, BCE, COMPOSITIONAL),
    # (POLYNOMIAL, L2, VTP),
    # (POLYNOMIAL, L2, TRILINEAR),
    # (POLYNOMIAL, L2, COMPOSITIONAL),
    # (POLYNOMIAL, MARGIN_LOSS_FN, VTP),
    # (POLYNOMIAL, MARGIN_LOSS_FN, TRILINEAR),
    # (POLYNOMIAL, MARGIN_LOSS_FN, COMPOSITIONAL),
    # (COMPLEX, BCE, VTP),
    # (COMPLEX, BCE, TRILINEAR),
    # (COMPLEX, BCE, COMPOSITIONAL),
    # (COMPLEX, L2, VTP),
    # (COMPLEX, L2, TRILINEAR),
    # (COMPLEX, L2, COMPOSITIONAL),
    # (COMPLEX, MARGIN_LOSS_FN, VTP),
    # (COMPLEX, MARGIN_LOSS_FN, TRILINEAR),
    # (COMPLEX, MARGIN_LOSS_FN, COMPOSITIONAL),
    (NEURALNETWORK, BCE, VTP),
    (NEURALNETWORK, BCE, TRILINEAR),
    (NEURALNETWORK, BCE, COMPOSITIONAL),
    (NEURALNETWORK, BCE, NEWSCORE),
    # (NEURALNETWORK, L2, VTP),
    # (NEURALNETWORK, L2, TRILINEAR),
    # (NEURALNETWORK, L2, COMPOSITIONAL),
    (NEURALNETWORK, MARGIN_LOSS_FN, VTP),
    (NEURALNETWORK, MARGIN_LOSS_FN, TRILINEAR),
    (NEURALNETWORK, MARGIN_LOSS_FN, COMPOSITIONAL),
    (NEURALNETWORK, MARGIN_LOSS_FN, NEWSCORE)
]

# def xavier_normal_init(embedding):
#     nn.init.xavier_normal_(embedding.weight.data)


# results = []
#
#
# def compute_MRR_batched(test_loader, all_entity_indices):
#     rr_sum = 0.0
#     total_triples = 0
#
#     for batch_triples in test_loader:
#         # Move batch_triples to the correct device
#         batch_triples = batch_triples.to(device)
#
#         h_batch = batch_triples[:, 0]
#         r_batch = batch_triples[:, 1]
#         t_true_batch = batch_triples[:, 2]
#
#         for h_idx, r_idx, t_true_idx in zip(h_batch, r_batch, t_true_batch):
#             scores = []
#
#             # Score all entities as potential tails for each (head, relation) pair
#             for t_idx in all_entity_indices:
#                 # Convert indices to tensors and ensure they are on the correct device
#                 h_tensor = torch.tensor([h_idx], device=device)
#                 r_tensor = torch.tensor([r_idx], device=device)
#                 t_tensor = torch.tensor([t_idx], device=device)
#
#                 score = SCORING_MAP[SCORING_FN](entity_embeddings(h_tensor),
#                                                 relation_embeddings(r_tensor),
#                                                 entity_embeddings(t_tensor)).item()
#                 scores.append((t_idx, score))
#
#             # Sort entities based on their scores
#             ranked_entities = sorted(scores, key=lambda x: x[1], reverse=True)
#             rank = [idx for idx, (entity, _) in enumerate(ranked_entities) if entity == t_true_idx.item()][0] + 1
#
#             # Add the reciprocal rank to the sum
#             rr_sum += 1.0 / rank
#             total_triples += 1
#
#     # Compute the mean reciprocal rank
#     mrr = rr_sum / total_triples
#     return mrr


test_dataset = TriplesDataset(test_triples_tensor)
test_loader = DataLoader(test_dataset, batch_size=32,drop_last=False, shuffle=False)
all_entity_indices = list(range(num_entities))
neural_network = NeuralNet(2, 50, 0.2, "tanh", 50 ).to(device)


# optimizer = optim.Adam([
#     {'params': entity_embeddings.parameters()},
#     {'params': relation_embeddings.parameters()},
#     {'params': neural_network.parameters()}
# ],
#     lr=0.001)

# Ensure the model is in evaluation mode

def compute_metrics_batched(test_loader, all_entity_indices, k_values=[1, 3, 10]):
    mrr_sum = 0.0
    hits_at_k_counts = {k: 0 for k in k_values}
    total_triples = 0

    with torch.no_grad():
        for batch_triples in test_loader:
            batch_triples = batch_triples.to(device)

            h_batch = batch_triples[:, 0]
            r_batch = batch_triples[:, 1]
            t_true_batch = batch_triples[:, 2]

            for h_idx, r_idx, t_true_idx in zip(h_batch, r_batch, t_true_batch):
                scores = []

                for t_idx in all_entity_indices:
                    h_tensor = torch.tensor([h_idx], device=device)
                    r_tensor = torch.tensor([r_idx], device=device)
                    t_tensor = torch.tensor([t_idx], device=device)

                    score = SCORING_MAP[SCORING_FN](entity_embeddings(h_tensor),
                                                    relation_embeddings(r_tensor),
                                                    entity_embeddings(t_tensor)).item()
                    scores.append((t_idx, score))

                ranked_entities = sorted(scores, key=lambda x: x[1], reverse=True)
                rank = [idx for idx, (entity, _) in enumerate(ranked_entities) if entity == t_true_idx.item()][0] + 1

                mrr_sum += 1.0 / rank

                for k in k_values:
                    hits_at_k_counts[k] += 1 if rank <= k else 0

                total_triples += 1

    mrr = mrr_sum / total_triples
    hits_at_k = {k: hits_at_k_counts[k] / total_triples for k in k_values}

    return mrr, hits_at_k


# Call the function after training your model

def initialize_weights(embedding):
    # nn.init.uniform_(embedding.weight.data, -0.05, 0.05)
    nn.init.xavier_uniform_(embedding.weight.data)


# for config in configurations:
#     initialize_weights(entity_embeddings)
#     initialize_weights(relation_embeddings)
#     configuration(*config)
#
#     optimizer = optim.Adam([
#         {'params': entity_embeddings.parameters()},
#         {'params': relation_embeddings.parameters()},
#         {'params': neural_network.parameters()}
#     ], lr=0.001)
#
#     # Example training loop with batches
#     for epoch in range(25):  # let's assume 2 epochs
#         total_loss = 0
#         for batch_triples in train_loader:
#             batch_loss = forward_batch(batch_triples, optimizer)
#             total_loss += batch_loss
#         # print(f"Epoch {epoch + 1}/100, Loss: {total_loss}")
#
#     # Evaluation
#     neural_network.eval()
#
#     # Disable gradient calculations
#     with torch.no_grad():
#         # Your MRR calculation code here
#         for batch_triples in test_loader:
#             mrr, hits_at_k = compute_metrics_batched(test_loader, all_entity_indices)
#             print(f'MRR:{config}, {mrr}')
#             for k, value in hits_at_k.items():
#                 print(f'Hits@{k}: {value}')
#     # mrr_value = compute_MRR_batched(test_loader, all_entity_indices)
#     # # Log the MRR
#     # logging.info(f'Mean Reciprocal Rank (MRR): {mrr_value}')
#     #
#     # print(f"Configuration: {config},(MRR) on Test Data: {mrr_value}")
#
#     # # Log the results
#     # results.append((config, mrr_value))

for config in configurations:
    # Initialize weights for each configuration
    initialize_weights(entity_embeddings)
    initialize_weights(relation_embeddings)

    # Apply the current configuration
    configuration(*config)

    # Initialize the optimizer with the current parameters
    optimizer = optim.Adam([
        {'params': entity_embeddings.parameters()},
        {'params': relation_embeddings.parameters()},
        {'params': neural_network.parameters()}
    ], lr=0.001)

    for config in configurations:
        # Initialize weights for each configuration
        initialize_weights(entity_embeddings)
        initialize_weights(relation_embeddings)

        # Apply the current configuration
        configuration(*config)

        # Initialize the optimizer with the current parameters
        optimizer = optim.Adam([
            {'params': entity_embeddings.parameters()},
            {'params': relation_embeddings.parameters()},
            {'params': neural_network.parameters()}
        ], lr=0.001)

        # Training loop
        neural_network.train()  # Set the network to training mode
        best_mrr = 0
        for epoch in range(10):  # Adjust the number of epochs as needed
            total_loss = 0
            for batch_triples in train_loader:
                optimizer.zero_grad()  # Clear previous gradients
                batch_loss = forward_k_vs_all(batch_triples)  # Forward pass and compute loss
                batch_loss.backward()  # Backpropagation
                optimizer.step()  # Update weights
                total_loss += batch_loss.item()  # Accumulate the total loss

            # Print training loss
            print(f"Epoch {epoch + 1}/25, Loss: {total_loss}")

        # Evaluation after training is done
        neural_network.eval()  # Set the network to evaluation mode
        with torch.no_grad():  # Disable gradient calculations
            mrr, hits_at_k = compute_metrics_batched(test_loader, all_entity_indices)

            # Check if this is the best MRR overall and save the model
            if mrr > best_mrr:
                best_mrr = mrr
                torch.save(neural_network.state_dict(), 'best_model.pth')

        # Print final evaluation metrics
        print(f'Final MRR for Configuration: {config}, MRR: {mrr}')
        for k, value in hits_at_k.items():
            print(f'Hits@{k}: {value}')
