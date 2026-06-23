"""
Data format definitions and validation for LLMRouter.

Defines abstract base classes and validators for:
1. Standard router data format (query-response pairs)
2. GMTRouter data format (JSONL with embeddings and ratings)

Usage Examples:

1. Automatic Format Detection:
    ```python
    from llmrouter.data import DataFormatDetector

    detector = DataFormatDetector()
    is_valid, format_type, error_msg = detector.validate_and_detect(data)

    if not is_valid:
        print(f"Validation error: {error_msg}")
    else:
        print(f"Detected format: {format_type}")
    ```

2. Validate GMTRouter Data:
    ```python
    from llmrouter.data import GMTRouterInteraction
    from pydantic import ValidationError

    try:
        interaction = GMTRouterInteraction(**data)
        print("Valid GMTRouter data!")
    except ValidationError as e:
        print(f"Invalid data: {e}")
    ```

3. Validate Standard Query Data:
    ```python
    from llmrouter.data import StandardQueryData

    query = StandardQueryData(
        query="What is machine learning?",
        task="qa",
        query_id="q_001"
    )
    ```

4. Get Format Requirements:
    ```python
    from llmrouter.data import get_format_requirements, DataFormatType

    requirements = get_format_requirements(DataFormatType.GMTROUTER)
    print(f"Required fields: {requirements['required_fields']}")
    print(f"Example: {requirements['example']}")
    ```
"""

from typing import Any, Dict, List, Optional
from abc import ABC, abstractmethod
from enum import Enum

from pydantic import BaseModel, Field, field_validator


class DataFormatType(Enum):
    """Supported data format types."""
    STANDARD = "standard"  # Standard LLMRouter format
    GMTROUTER = "gmtrouter"  # GMTRouter special JSONL format
    UNKNOWN = "unknown"


# ============================================================================
# Abstract Base Classes for Router Data Formats
# ============================================================================

class RouterDataFormat(ABC):
    """
    Abstract base class for router data formats.

    All router data formats must implement:
    - validate(): Validate data structure
    - get_format_type(): Return format type
    """

    @abstractmethod
    def validate(self, data: Any) -> bool:
        """
        Validate data conforms to this format.

        Args:
            data: Data to validate

        Returns:
            bool: True if valid, False otherwise
        """
        pass

    @abstractmethod
    def get_format_type(self) -> DataFormatType:
        """Return the format type."""
        pass

    @abstractmethod
    def get_required_fields(self) -> List[str]:
        """Return list of required fields."""
        pass


# ============================================================================
# Standard Router Data Format (Query-Response Pairs)
# ============================================================================

class StandardQueryData(BaseModel):
    """
    Standard query data format for regular routers.

    Format:
    {
      "query": "What is machine learning?",
      "task": "qa",
      "query_id": "q_001"  # optional
    }
    """
    query: str = Field(..., description="Query text")
    task: Optional[str] = Field(default=None, description="Task type (qa, math, code, etc.)")
    query_id: Optional[str] = Field(default=None, description="Query identifier")

    @field_validator('query')
    @classmethod
    def query_not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('Query cannot be empty')
        return v


class StandardRoutingData(BaseModel):
    """
    Standard routing data format (training labels).

    Format:
    {
      "query_id": "q_001",
      "best_model": "gpt-4",
      "model_scores": {"gpt-4": 0.95, "gpt-3.5": 0.82}  # optional
    }
    """
    query_id: str = Field(..., description="Query identifier")
    best_model: str = Field(..., description="Best model for this query")
    model_scores: Optional[Dict[str, float]] = Field(default=None, description="Model performance scores")


class StandardDataFormat(RouterDataFormat):
    """Standard router data format validator."""

    def validate(self, data: Dict) -> bool:
        """Validate standard format."""
        try:
            if isinstance(data, dict):
                # Check if it's query data
                if 'query' in data:
                    StandardQueryData(**data)
                    return True
                # Check if it's routing data
                elif 'query_id' in data and 'best_model' in data:
                    StandardRoutingData(**data)
                    return True
            return False
        except Exception:
            return False

    def get_format_type(self) -> DataFormatType:
        return DataFormatType.STANDARD

    def get_required_fields(self) -> List[str]:
        return ["query"]  # Minimum requirement


# ============================================================================
# GMTRouter Data Format (JSONL with Embeddings and Ratings)
# ============================================================================

class GMTRouterConversationTurn(BaseModel):
    """
    Single turn in GMTRouter conversation.

    Format:
    {
      "query": "What is machine learning?",
      "query_emb": [0.1, 0.2, 0.3, ...],  # Query embedding from PLM
      "response": "Machine learning is...",  # optional
      "rating": 4.5  # Quality rating (0-5)
    }
    """
    query: str = Field(..., description="Query text")
    query_emb: List[float] = Field(..., description="Query embedding vector from PLM")
    response: Optional[str] = Field(default=None, description="Response text")
    rating: float = Field(..., description="Quality rating (0-5 or 0-1)")

    @field_validator('query_emb')
    @classmethod
    def validate_embedding(cls, v):
        if not v or len(v) == 0:
            raise ValueError('Query embedding cannot be empty')
        return v

    @field_validator('rating')
    @classmethod
    def validate_rating(cls, v):
        if not (0 <= v <= 5):
            raise ValueError('Rating must be between 0 and 5')
        return v


class GMTRouterInteraction(BaseModel):
    """
    GMTRouter interaction data format.

    Format:
    {
      "judge": "user_001",  # User identifier
      "model": "gpt-4",  # LLM model name
      "question_id": "q_12345",  # Question identifier
      "turn": 1,  # Turn number in conversation
      "conversation": [  # List of conversation turns
        {
          "query": "What is ML?",
          "query_emb": [0.1, 0.2, ...],
          "response": "ML is...",
          "rating": 4.5
        }
      ],
      "model_emb": [0.3, 0.4, ...],  # LLM embedding vector
      "encoder": "sentence-transformers/all-mpnet-base-v2"  # optional
    }
    """
    judge: str = Field(..., description="User identifier (e.g., 'user_001')")
    model: str = Field(..., description="LLM model name (e.g., 'gpt-4')")
    question_id: str = Field(..., description="Question/task identifier")
    turn: int = Field(..., description="Turn number in multi-turn conversation")
    conversation: List[GMTRouterConversationTurn] = Field(..., description="Conversation turns")
    model_emb: List[float] = Field(..., description="LLM embedding vector from PLM")
    encoder: Optional[str] = Field(default=None, description="PLM model name used for embeddings")

    @field_validator('conversation')
    @classmethod
    def validate_conversation(cls, v):
        if not v or len(v) == 0:
            raise ValueError('Conversation cannot be empty')
        return v

    @field_validator('model_emb')
    @classmethod
    def validate_model_emb(cls, v):
        if not v or len(v) == 0:
            raise ValueError('Model embedding cannot be empty')
        return v

    @field_validator('turn')
    @classmethod
    def validate_turn(cls, v):
        if v < 1:
            raise ValueError('Turn must be >= 1')
        return v


class GMTRouterDataFormat(RouterDataFormat):
    """GMTRouter data format validator."""

    def validate(self, data: Dict) -> bool:
        """Validate GMTRouter format."""
        try:
            GMTRouterInteraction(**data)
            return True
        except Exception:
            return False

    def get_format_type(self) -> DataFormatType:
        return DataFormatType.GMTROUTER

    def get_required_fields(self) -> List[str]:
        return ["judge", "model", "question_id", "turn", "conversation", "model_emb"]


# ============================================================================
# Data Format Detection
# ============================================================================

class DataFormatDetector:
    """
    Automatic data format detector.

    Detects whether data is:
    - Standard router format
    - GMTRouter format
    - Unknown format
    """

    def __init__(self):
        self.standard_validator = StandardDataFormat()
        self.gmtrouter_validator = GMTRouterDataFormat()

    def detect_format(self, data: Dict) -> DataFormatType:
        """
        Detect data format type.

        Args:
            data: Data dictionary to detect

        Returns:
            DataFormatType: Detected format type
        """
        # Check GMTRouter format first (more specific)
        if self.gmtrouter_validator.validate(data):
            return DataFormatType.GMTROUTER

        # Check standard format
        if self.standard_validator.validate(data):
            return DataFormatType.STANDARD

        # Unknown format
        return DataFormatType.UNKNOWN

    def validate_and_detect(self, data: Dict) -> tuple:
        """
        Validate and detect data format.

        Args:
            data: Data to validate

        Returns:
            Tuple of (is_valid, format_type, error_message)
        """
        format_type = self.detect_format(data)

        if format_type == DataFormatType.UNKNOWN:
            return False, format_type, "Unknown data format. Must be standard or GMTRouter format."

        # Validate based on detected format
        try:
            if format_type == DataFormatType.GMTROUTER:
                GMTRouterInteraction(**data)
            else:
                # Try both query and routing data
                if 'query' in data:
                    StandardQueryData(**data)
                else:
                    StandardRoutingData(**data)
            return True, format_type, None
        except Exception as e:
            return False, format_type, str(e)


# ============================================================================
# Utility Functions
# ============================================================================

def get_format_requirements(format_type: DataFormatType) -> Dict[str, Any]:
    """
    Get format requirements and examples.

    Args:
        format_type: Data format type

    Returns:
        Dictionary with requirements and examples
    """
    if format_type == DataFormatType.STANDARD:
        return {
            "name": "Standard Router Format",
            "description": "Standard query-response format for regular routers",
            "required_fields": ["query"],
            "optional_fields": ["task", "query_id"],
            "example_query": {
                "query": "What is machine learning?",
                "task": "qa",
                "query_id": "q_001"
            },
            "example_routing": {
                "query_id": "q_001",
                "best_model": "gpt-4",
                "model_scores": {"gpt-4": 0.95, "gpt-3.5": 0.82}
            }
        }
    elif format_type == DataFormatType.GMTROUTER:
        return {
            "name": "GMTRouter Format",
            "description": "Special JSONL format with embeddings and ratings for personalized routing",
            "required_fields": ["judge", "model", "question_id", "turn", "conversation", "model_emb"],
            "optional_fields": ["encoder"],
            "example": {
                "judge": "user_001",
                "model": "gpt-4",
                "question_id": "q_12345",
                "turn": 1,
                "conversation": [
                    {
                        "query": "What is ML?",
                        "query_emb": [0.123, -0.456, 0.789],
                        "response": "ML is...",
                        "rating": 4.5
                    }
                ],
                "model_emb": [0.234, -0.567, 0.891],
                "encoder": "sentence-transformers/all-mpnet-base-v2"
            },
            "note": "Embeddings should come from a pre-trained language model (PLM)"
        }
    else:
        return {
            "name": "Unknown Format",
            "description": "Format not recognized",
            "error": "Please use standard or GMTRouter format"
        }


def print_format_help(format_type: Optional[DataFormatType] = None):
    """
    Print help information about data formats.

    Args:
        format_type: Specific format to show help for (None for all)
    """
    if format_type is None:
        # Show all formats
        print("=" * 70)
        print("LLMRouter Data Formats")
        print("=" * 70)
        print()

        for fmt in [DataFormatType.STANDARD, DataFormatType.GMTROUTER]:
            info = get_format_requirements(fmt)
            print(f"Format: {info['name']}")
            print(f"Description: {info['description']}")
            print(f"Required fields: {', '.join(info['required_fields'])}")
            print(f"Optional fields: {', '.join(info.get('optional_fields', []))}")
            print()
    else:
        # Show specific format
        info = get_format_requirements(format_type)
        print("=" * 70)
        print(f"Format: {info['name']}")
        print("=" * 70)
        print(f"Description: {info['description']}")
        print(f"\nRequired fields: {', '.join(info['required_fields'])}")
        if 'optional_fields' in info:
            print(f"Optional fields: {', '.join(info['optional_fields'])}")

        if 'example' in info:
            import json
            print(f"\nExample:")
            print(json.dumps(info['example'], indent=2))
        elif 'example_query' in info:
            import json
            print(f"\nExample Query:")
            print(json.dumps(info['example_query'], indent=2))
            if 'example_routing' in info:
                print(f"\nExample Routing Data:")
                print(json.dumps(info['example_routing'], indent=2))
