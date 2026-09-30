"""Movie Recommendation System: content-based, collaborative and hybrid."""
from .data_loader import load_data
from .content_based import ContentRecommender
from .collaborative import CollaborativeRecommender
from .hybrid import HybridRecommender

__all__ = [
    "load_data",
    "ContentRecommender",
    "CollaborativeRecommender",
    "HybridRecommender",
]
__version__ = "1.0.0"
