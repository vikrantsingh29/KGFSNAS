
import torch
from kgfsnas import *


def test_polynomial():
    # Test Case 1: Simple evaluation
    embedding = torch.tensor([2, 3, 1])  # Assume it's a quadratic polynomial: 2x^2 + 3x + 1
    x_samples = torch.tensor([0, 1, 2])
    output = polynomial(embedding, x_samples)
    expected_output = torch.tensor([2,6,12])
    assert torch.allclose(output, expected_output), f"Expected {expected_output}, but got {output}"

    # Test Case 2: Edge case with zero embedding
    embedding = torch.tensor([0, 0, 0])
    x_samples = torch.tensor([0, 1, 2])
    output = polynomial(embedding, x_samples)
    expected_output = torch.tensor([0, 0, 0])
    assert torch.allclose(output, expected_output), f"Expected {expected_output}, but got {output}"

    # Test Case 3: Another random evaluation
    embedding = torch.tensor([1, -2, 3])
    x_samples = torch.tensor([-1, 0, 1])
    output = polynomial(embedding, x_samples)
    expected_output = torch.tensor([6, 1, 2])
    assert torch.allclose(output, expected_output), f"Expected {expected_output}, but got {output}"

def test_calculate_loss_BCELogistLoss():
    # Test Case 1: Typical scores
    pos_scores = torch.tensor([2.5, 1.5, -1.5])
    neg_scores = torch.tensor([-2.5, -1.5, 1.5])
    loss = calculate_loss_BCELogistLoss(pos_scores, neg_scores)
    assert isinstance(loss, torch.Tensor) and loss.dim() == 0

    # Test Case 2: Zero scores
    pos_scores = torch.tensor([0, 0, 0])
    neg_scores = torch.tensor([0, 0, 0])
    loss = calculate_loss_BCELogistLoss(pos_scores, neg_scores)
    assert isinstance(loss, torch.Tensor) and loss.dim() == 0

    # Test Case 3: Mixed scores
    pos_scores = torch.tensor([-1, 1, 0])
    neg_scores = torch.tensor([1, -1, 0])
    loss = calculate_loss_BCELogistLoss(pos_scores, neg_scores)
    assert isinstance(loss, torch.Tensor) and loss.dim() == 0

def test_generate_negative_triples():
    # Test Case 1: Typical triples
    triple = (0, 1, 2)
    all_entities = torch.arange(10)
    num_samples = 5
    negative_triples = generate_negative_triples(triple, all_entities, num_samples)
    assert len(negative_triples) == num_samples

    # Test Case 2: Edge case with zero samples
    triple = (0, 1, 2)
    all_entities = torch.arange(10)
    num_samples = 0
    negative_triples = generate_negative_triples(triple, all_entities, num_samples)
    assert len(negative_triples) == num_samples

    # Test Case 3: Edge case with zero entities
    triple = (0, 1, 2)
    all_entities = torch.tensor([])
    num_samples = 5
    negative_triples = generate_negative_triples(triple, all_entities, num_samples)
    assert len(negative_triples) == num_samples


