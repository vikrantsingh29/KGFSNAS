import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn import functional as F
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
dummy_triples = train_triples
true_triples_set = set(map(tuple, dummy_triples))

# Convert triples to tensor
triples_tensor = torch.tensor(dummy_triples)
test_triples_tensor = torch.tensor(test_triples)


# Function spaces
def complex_num_functionSpace_batch(embedding, x_samples_batch):
    # Prepare x_samples for broadcasting
    # x_samples_batch = x_samples.unsqueeze(0).expand(embedding.size(0), -1)

    # Compute the real part for each dimension
    real_part = embedding * torch.cos(x_samples_batch)
    # Compute the imaginary part for each dimension
    imag_part = embedding * torch.sin(x_samples_batch)

    # Sum the real and imaginary parts
    complex_representation = real_part + imag_part

    return complex_representation


def polynomial_functionSpace_batch(embedding, x_samples):
    # The powers tensor will have shape [batch_size, embedding_dim, embedding_dim]
    powers_of_x = torch.stack([x_samples ** i for i in range(embedding.size(1))], dim=-1)

    # Ensure that embedding is broadcastable over the powers tensor for element-wise multiplication
    # The embedding tensor will have shape [batch_size, 1, embedding_dim]
    emb_expanded = embedding.unsqueeze(1)

    # Element-wise multiplication of the embedding vectors with the powers of x
    poly_vector = emb_expanded * powers_of_x

    # Sum along the last dimension to get the final polynomial vector for each embedding
    # The resulting tensor will have shape [batch_size, embedding_dim]
    poly_vector = poly_vector.sum(dim=-1)

    # Normalize the output vector for each item in the batch
    poly_vector = F.normalize(poly_vector, p=2, dim=-1)

    return poly_vector

class NeuralNet(nn.Module):
    def __init__(self, num_layers, layer_size, dropout_rate, activation_function, input_dim):
        super().__init__()

        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout_rate)
        self.activation_function = activation_function

        # Start with double the input_dim because we concatenate input_embedding and x_samples
        current_dim = 2 * input_dim
        for i in range(num_layers - 1):
            self.layers.append(nn.Linear(current_dim, layer_size))
            current_dim = layer_size

        self.layers.append(nn.Linear(current_dim, input_dim))

    def forward(self, input_embedding, x_samples):
        # Concatenate input_embedding with x_samples along the last dimension
        x = torch.cat((input_embedding, x_samples), dim=-1)

        for layer in self.layers[:-1]:
            x = layer(x)
            if self.activation_function == 'relu':
                x = torch.relu(x)
            elif self.activation_function == 'tanh':
                x = torch.tanh(x)
            elif self.activation_function == 'sigmoid':
                x = torch.sigmoid(x)
            x = self.dropout(x)

        x = self.layers[-1](x)
        return x

# tuned_params = nni.get_next_parameter()
# num_layers = tuned_params['num_layers']
# layer_size = tuned_params['layer_size']
# dropout_rate = tuned_params['dropout_rate']
# activation_function = tuned_params['activation_function']
# learning_rate = tuned_params['learning_rate']
# num_epochs = tuned_params['num_epochs']
# batch_size = tuned_params['batch_size']
# weight_decay = tuned_params['weight_decay']
# input_dropout_rate = tuned_params['input_dropout_rate']
# hidden_dropout_rate = tuned_params['hidden_dropout_rate']
# feature_map_dropout_rate = tuned_params['feature_map_dropout_rate']
# normalization = tuned_params['normalization']
# init_param = tuned_params['init_param']
# gradient_accumulation_steps = tuned_params['gradient_accumulation_steps']
# num_folds_for_cv = tuned_params['num_folds_for_cv']
# eval_model = tuned_params['eval_model']
# save_model_at_every_epoch = tuned_params['save_model_at_every_epoch']
# label_smoothing_rate = tuned_params['label_smoothing_rate']
# kernel_size = tuned_params['kernel_size']
# num_of_output_channels = tuned_params['num_of_output_channels']
# num_core = tuned_params['num_core']
# random_seed = tuned_params['random_seed']
# sample_triples_ratio = tuned_params['sample_triples_ratio']
# read_only_few = tuned_params['read_only_few']
# add_noise_rate = tuned_params['add_noise_rate']

# Scoring functions
def compute_compostional_score_batch(h, r, t):
    # Assume x_samples are pre-computed and sent to the device
    x_samples = torch.linspace(0, 1, steps=embedding_dim).to(h.device)
    x_samples_batch = x_samples.unsqueeze(0).expand(h.size(0), -1)
    # Compute function space representations
    fh = polynomial_functionSpace_batch(h, x_samples_batch)
    frfh = polynomial_functionSpace_batch(r, fh)
    ft = polynomial_functionSpace_batch(t, x_samples_batch)

    # Here we perform element-wise multiplication across the batch
    score = torch.trapz(frfh * ft, x_samples, dim=1)

    return score


# Loss functions
def compute_margin_loss_batch(positive_scores, negative_scores, k):
    # Reshape positive_scores to compare against each negative score
    positive_scores_repeated = positive_scores.unsqueeze(1).repeat(1, k)

    # Calculate the margin ranking loss for each positive-negative pair
    loss = F.margin_ranking_loss(
        positive_scores_repeated.view(-1),
        negative_scores.view(-1),
        torch.ones(positive_scores_repeated.numel()),
        margin=1.0,
        reduction='none'
    )

    # Average the loss over the number of negatives per positive
    loss = loss.view(-1, k).mean(dim=1).mean()
    return loss


# DataLoader
class TriplesDataset(Dataset):
    def __init__(self, triples):
        self.triples = triples

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        return self.triples[idx]


train_dataset = TriplesDataset(triples_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Mock embedding layers
# entity_embeddings = torch.nn.Embedding(num_entities, embedding_dim)
# relation_embeddings = torch.nn.Embedding(num_relations, embedding_dim)
entity_embeddings = nn.Embedding(num_entities, embedding_dim).to(device)
relation_embeddings = nn.Embedding(num_relations, embedding_dim).to(device)

optimizer = optim.Adam([
    {'params': entity_embeddings.parameters()},
    {'params': relation_embeddings.parameters()}
], lr=0.001)


def generate_negative_triples_batch(positive_triples_batch, num_entities, true_triples_set, k=5):
    negative_triples_batch = []

    for h, r, t in positive_triples_batch.numpy():
        negatives_for_this_triple = 0
        while negatives_for_this_triple < k:
            # Randomly select a negative tail entity that is not the true tail
            # and the (head, relation, negative_tail) does not exist in the true triples set
            negative_tail = torch.randint(0, num_entities, (1,)).item()
            if negative_tail != t and (h, r, negative_tail) not in true_triples_set:
                negative_triples_batch.append((h, r, negative_tail))
                negatives_for_this_triple += 1

    return torch.tensor(negative_triples_batch)


def forward_batch(triples_batch):
    # Get positive embeddings

    triples_batch = triples_batch.to(device)
    h_embeddings = entity_embeddings(triples_batch[:, 0])
    r_embeddings = relation_embeddings(triples_batch[:, 1])
    t_embeddings = entity_embeddings(triples_batch[:, 2])

    # Compute scores for positive triples
    positive_scores = compute_compostional_score_batch(h_embeddings, r_embeddings, t_embeddings)

    # Generate negative triples for the batch, k negative samples per positive sample
    negative_triples_batch = generate_negative_triples_batch(triples_batch, num_entities, true_triples_set, k=5)

    # Get negative embeddings
    h_negative_embeddings = entity_embeddings(negative_triples_batch[:, 0])
    r_negative_embeddings = relation_embeddings(negative_triples_batch[:, 1])
    t_negative_embeddings = entity_embeddings(negative_triples_batch[:, 2])

    # Compute scores for negative triples
    negative_scores = compute_compostional_score_batch(h_negative_embeddings, r_negative_embeddings,
                                                       t_negative_embeddings)

    # Compute batch loss using the custom margin loss function
    batch_loss = compute_margin_loss_batch(positive_scores, negative_scores, k=5)

    # Backpropagate and update weights
    optimizer.zero_grad()
    batch_loss.backward()
    optimizer.step()

    return batch_loss.item()


# Example training loop with batches
for epoch in range(100):  # let's assume 2 epochs
    total_loss = 0
    for batch_triples in train_loader:
        batch_loss = forward_batch(batch_triples)
        total_loss += batch_loss
    print(f"Epoch {epoch + 1}/100, Loss: {total_loss}")


def compute_MRR_batched(test_loader, all_entity_indices):
    all_entity_indices = torch.tensor(all_entity_indices).to(device)

    rr_sum = 0.0
    total_triples = 0

    for batch_triples in test_loader:
        h_batch = batch_triples[:, 0]
        r_batch = batch_triples[:, 1]
        t_true_batch = batch_triples[:, 2]

        for h_idx, r_idx, t_true_idx in zip(h_batch, r_batch, t_true_batch):
            scores = []

            # Score all entities as potential tails for each (head, relation) pair
            for t_idx in all_entity_indices:
                # Assuming compute_compostional_score_batch can handle single samples as well
                score = compute_compostional_score_batch(entity_embeddings(torch.tensor([h_idx])),
                                                         relation_embeddings(torch.tensor([r_idx])),
                                                         entity_embeddings(torch.tensor([t_idx]))).item()
                scores.append((t_idx, score))

            # Sort entities based on their scores
            ranked_entities = sorted(scores, key=lambda x: x[1], reverse=True)
            rank = [idx for idx, (entity, _) in enumerate(ranked_entities) if entity == t_true_idx.item()][0] + 1

            # Add the reciprocal rank to the sum
            rr_sum += 1.0 / rank
            total_triples += 1

    # Compute the mean reciprocal rank
    mrr = rr_sum / total_triples
    return mrr


test_dataset = TriplesDataset(test_triples_tensor)  # Assuming test_triples_tensor is defined
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
all_entity_indices = list(range(num_entities))

mrr = compute_MRR_batched(test_loader, all_entity_indices)
print("Mean Reciprocal Rank (MRR):", mrr)
