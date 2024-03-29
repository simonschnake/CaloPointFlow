import torch
from torch import LongTensor, Tensor
from torch.nested import nested_tensor

def to_nested(flat_tensor: Tensor, nnz: LongTensor) -> Tensor:
    start_index = nnz.cumsum(dim=0)[:-1]
    ts = torch.tensor_split(flat_tensor, start_index)
    return nested_tensor(list(ts))

def to_flat(nested_tensor: nested_tensor) -> Tensor:
    if not nested_tensor.is_nested:
        raise ValueError("Not a nested tensor")
    
    ft = torch.cat(nested_tensor.unbind())

    return ft

if __name__ == "__main__":
    flat_tensor = torch.randn(100, 5, 3)
    nnz = torch.tensor([10, 20, 30, 40])
    nested_tensor = to_nested(flat_tensor, nnz)
    assert torch.allclose(flat_tensor, to_flat(nested_tensor))