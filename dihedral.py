import torch

to = {
    0: lambda ts: ts,
    1: lambda ts: torch.rot90(ts, 1, (-2, -1)),
    2: lambda ts: torch.rot90(ts, 2, (-2, -1)),
    3: lambda ts: torch.rot90(ts, 3, (-2, -1)),
    4: lambda ts: torch.flip(ts, (-1,)),
    5: lambda ts: torch.flip(ts, (-2,)),
    6: lambda ts: ts.transpose(-2, -1),
    7: lambda ts: torch.flip(ts.transpose(-2, -1), (-2, -1)),
}

reverse = {
    0: 0,
    1: 3,
    2: 2,
    3: 1,
    4: 4,
    5: 5,
    6: 6,
    7: 7,
}
