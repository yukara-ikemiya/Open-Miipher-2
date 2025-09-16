"""
Copyright (C) 2025 Yukara Ikemiya
"""
from abc import ABC, abstractmethod


class Degradation(ABC):
    def __init__(self, sample_rate: int = 16000):
        super().__init__()
        self.sample_rate = sample_rate

    @abstractmethod
    def __call__(self, x):
        pass
