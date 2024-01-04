import pytest
import torch
from mobilevlm import LDP  # replace with your actual model class


@pytest.fixture
def ldp():
    return LDP(in_channels=128, out_channels=128)


def test_forward(ldp):
    for i in range(1, 21):
        input_tensor = torch.rand(1, 128, i, i)
        output = ldp.forward(input_tensor)
        assert output.shape == (
            1,
            128,
            i // 2,
            i // 2,
        )  # Expected output shape


def test_zero_input(ldp):
    input_tensor = torch.zeros(1, 128, 64, 64)
    output = ldp.forward(input_tensor)
    assert (output == 0).all()


def test_large_input(ldp):
    input_tensor = torch.rand(1, 128, 1024, 1024)
    output = ldp.forward(input_tensor)
    assert output.shape == (1, 128, 512, 512)  # Expected output shape


def test_small_input(ldp):
    input_tensor = torch.rand(1, 128, 1, 1)
    output = ldp.forward(input_tensor)
    assert output.shape == (1, 128, 1, 1)  # Expected output shape


def test_negative_input(ldp):
    input_tensor = -torch.rand(1, 128, 64, 64)
    output = ldp.forward(input_tensor)
    assert (output <= 0).all()  # Assuming ReLU or similar activation
