from dataclasses import dataclass

@dataclass
class Config:
    min_word_count: int
    random_state: int
