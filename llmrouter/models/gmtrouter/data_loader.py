"""
Data loader for GMTRouter with format detection and heterogeneous graph construction.

GMTRouter uses a special JSONL format with user interactions, ratings, and embeddings.
This module handles loading, validation, and graph construction.
"""

import json
import torch
import numpy as np
from typing import Dict, List, Tuple, Any
from collections import defaultdict
from enum import Enum
from pydantic import ValidationError

# Import validation classes
from llmrouter.data import (
    DataFormatDetector,
    GMTRouterInteraction,
    DataFormatType,
)


class EmbInit(Enum):
    """Embedding initialization strategies for different node types."""
    ZEROS = "zeros"
    ONES = "ones"
    RANDOM = "random"
    PRETRAINED = "pretrained"


class HeteroEdges(Enum):
    """Heterogeneous edge types in the GMTRouter graph."""
    # User-Session edges
    USER_OWN_SESSION = "user_own_session"
    SESSION_OWNED_BY_USER = "session_owned_by_user"

    # Session-Query edges
    SESSION_HAS_QUERY = "session_has_query"
    QUERY_IN_SESSION = "query_in_session"

    # Query-Response edges
    QUERY_ANSWERED_BY_RESPONSE = "query_answered_by_response"
    RESPONSE_ANSWERED_TO_QUERY = "response_answered_to_query"

    # LLM-Response edges
    LLM_GENERATE_RESPONSE = "llm_generate_response"
    RESPONSE_GENERATED_BY_LLM = "response_generated_by_llm"

    # Temporal edges (next/prev)
    SESSION_NEXT = "session_next"
    SESSION_PREV = "session_prev"
    QUERY_NEXT = "query_next"
    QUERY_PREV = "query_prev"

    # Additional edges
    USER_PREFER_LLM = "user_prefer_llm"
    LLM_PREFERRED_BY_USER = "llm_preferred_by_user"
    QUERY_SIMILAR_TO_QUERY = "query_similar_to_query"
    LLM_SIMILAR_TO_LLM = "llm_similar_to_llm"

    # Response quality edges
    RESPONSE_HIGH_QUALITY = "response_high_quality"
    RESPONSE_LOW_QUALITY = "response_low_quality"


def detect_data_format(file_path: str) -> str:
    """
    Detect whether the data file is GMTRouter format or standard LLMRouter format.

    Uses the centralized DataFormatDetector for consistent format detection.

    Args:
        file_path: Path to the data file

    Returns:
        str: "gmtrouter" or "standard"
    """
    try:
        detector = DataFormatDetector()

        with open(file_path, 'r', encoding='utf-8') as f:
            first_line = f.readline().strip()
            if not first_line:
                return "standard"

            data = json.loads(first_line)

            # Use centralized format detector
            format_type = detector.detect_format(data)

            if format_type == DataFormatType.GMTROUTER:
                return "gmtrouter"
            elif format_type == DataFormatType.STANDARD:
                return "standard"
            else:
                # Unknown format defaults to standard
                print(f"Warning: Unknown format detected in {file_path}, defaulting to standard")
                return "standard"

    except Exception as e:
        print(f"Warning: Could not detect format for {file_path}: {e}")
        return "standard"


class GMTRouterDataLoader:
    """
    Data loader for GMTRouter with heterogeneous graph construction.

    Handles:
    - Loading JSONL files with user interactions
    - Building heterogeneous graph with 5 node types
    - Creating 21 edge types
    - Managing user/session/query/llm/response mappings
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize GMTRouter data loader.

        Args:
            config: Configuration dictionary with data paths and settings
        """
        self.config = config
        self.gmt_config = config.get("gmt_config", {})

        # Node mappings (string -> integer ID)
        self.user_map = {}
        self.session_map = {}
        self.query_map = {}
        self.llm_map = {}
        self.response_map = {}

        # Reverse mappings
        self.user_id_to_name = {}
        self.llm_id_to_name = {}

        # Embeddings
        self.query_embeddings = []
        self.llm_embeddings = []

        # Graph data
        self.edges = defaultdict(list)  # edge_type -> [(src, dst), ...]
        self.interactions = []  # List of (user, session, query, llm, response, rating)

        # Metadata
        self.num_users = 0
        self.num_sessions = 0
        self.num_queries = 0
        self.num_llms = 0
        self.num_responses = 0

        # Embedding dimension (will be set when loading data)
        self.embedding_dim = None

    def load_data(self, file_path: str) -> Tuple[Any, Any]:
        """
        Load GMTRouter JSONL data and construct heterogeneous graph.

        Args:
            file_path: Path to JSONL file

        Returns:
            tuple: (graph_data, metadata)
        """
        print(f"Loading GMTRouter data from {file_path}...")

        # Detect format
        format_type = detect_data_format(file_path)
        if format_type != "gmtrouter":
            raise ValueError(
                f"Data format mismatch! Expected GMTRouter format but got {format_type}.\n"
                f"GMTRouter requires JSONL with fields: judge, model, conversation, model_emb, turn\n"
                f"See llmrouter/models/gmtrouter/README.md for data format specification."
            )

        # Load JSONL with Pydantic validation
        interactions = []
        validation_errors = 0
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line.strip())

                    # Validate using Pydantic model
                    try:
                        validated_interaction = GMTRouterInteraction(**data)
                        # Convert back to dict for processing
                        interactions.append(validated_interaction.model_dump())
                    except ValidationError as ve:
                        validation_errors += 1
                        print(f"Warning: Validation error at line {line_num}:")
                        print(f"  {ve}")
                        if validation_errors > 10:
                            print(f"Too many validation errors (>{validation_errors}). Please check data format.")
                            print(f"See GMTRouter README for correct JSONL format.")
                            raise ValueError("Data validation failed")
                        continue

                except json.JSONDecodeError as e:
                    print(f"Warning: Skipping invalid JSON at line {line_num}: {e}")
                    continue

        if validation_errors > 0:
            print(f"Total validation errors: {validation_errors} lines skipped")

        print(f"Loaded {len(interactions)} interactions")

        # Build graph
        self._build_graph_from_interactions(interactions)

        # Create PyTorch Geometric data
        graph_data = self._create_hetero_data()

        # Create metadata for training
        metadata = self._create_metadata()

        return graph_data, metadata

    def _build_graph_from_interactions(self, interactions: List[Dict]):
        """
        Build heterogeneous graph from interaction data.

        Args:
            interactions: List of interaction dictionaries
        """
        print("Building heterogeneous graph...")

        # Track previous sessions/queries for temporal edges
        user_last_session = {}
        session_last_query = {}

        for interaction in interactions:
            user_name = interaction['judge']
            model_name = interaction['model']
            question_id = interaction.get('question_id', 'unknown')
            turn = interaction.get('turn', 1)
            conversation = interaction['conversation']
            model_emb = np.array(interaction.get('model_emb', []))

            # Get or create user ID
            if user_name not in self.user_map:
                self.user_map[user_name] = self.num_users
                self.user_id_to_name[self.num_users] = user_name
                self.num_users += 1
            user_id = self.user_map[user_name]

            # Get or create LLM ID
            if model_name not in self.llm_map:
                self.llm_map[model_name] = self.num_llms
                self.llm_id_to_name[self.num_llms] = model_name
                self.llm_embeddings.append(model_emb)
                self.num_llms += 1
            llm_id = self.llm_map[model_name]

            # Create session ID (user + question_id)
            session_key = f"{user_name}_{question_id}"
            if session_key not in self.session_map:
                self.session_map[session_key] = self.num_sessions

                # User-Session edges
                self.edges[HeteroEdges.USER_OWN_SESSION.value].append((user_id, self.num_sessions))
                self.edges[HeteroEdges.SESSION_OWNED_BY_USER.value].append((self.num_sessions, user_id))

                # Temporal session edge (if user has previous session)
                if user_name in user_last_session:
                    prev_session = user_last_session[user_name]
                    self.edges[HeteroEdges.SESSION_NEXT.value].append((prev_session, self.num_sessions))
                    self.edges[HeteroEdges.SESSION_PREV.value].append((self.num_sessions, prev_session))

                user_last_session[user_name] = self.num_sessions
                self.num_sessions += 1

            session_id = self.session_map[session_key]

            # Process each turn in conversation
            for conv_turn in conversation:
                query_text = conv_turn['query']
                query_emb = np.array(conv_turn.get('query_emb', []))
                rating = conv_turn.get('rating', 0.0)

                # Set embedding dimension
                if self.embedding_dim is None and len(query_emb) > 0:
                    self.embedding_dim = len(query_emb)

                # Create query ID
                query_key = f"{question_id}_{turn}_{query_text[:50]}"
                if query_key not in self.query_map:
                    self.query_map[query_key] = self.num_queries
                    self.query_embeddings.append(query_emb)

                    # Session-Query edges
                    self.edges[HeteroEdges.SESSION_HAS_QUERY.value].append((session_id, self.num_queries))
                    self.edges[HeteroEdges.QUERY_IN_SESSION.value].append((self.num_queries, session_id))

                    # Temporal query edge
                    if session_key in session_last_query:
                        prev_query = session_last_query[session_key]
                        self.edges[HeteroEdges.QUERY_NEXT.value].append((prev_query, self.num_queries))
                        self.edges[HeteroEdges.QUERY_PREV.value].append((self.num_queries, prev_query))

                    session_last_query[session_key] = self.num_queries
                    self.num_queries += 1

                query_id = self.query_map[query_key]

                # Create response ID
                response_id = self.num_responses

                # Query-Response edges
                self.edges[HeteroEdges.QUERY_ANSWERED_BY_RESPONSE.value].append((query_id, response_id))
                self.edges[HeteroEdges.RESPONSE_ANSWERED_TO_QUERY.value].append((response_id, query_id))

                # LLM-Response edges
                self.edges[HeteroEdges.LLM_GENERATE_RESPONSE.value].append((llm_id, response_id))
                self.edges[HeteroEdges.RESPONSE_GENERATED_BY_LLM.value].append((response_id, llm_id))

                # User-LLM preference edge (based on rating)
                if rating >= 4.0:  # High rating threshold
                    self.edges[HeteroEdges.USER_PREFER_LLM.value].append((user_id, llm_id))
                    self.edges[HeteroEdges.LLM_PREFERRED_BY_USER.value].append((llm_id, user_id))

                # Response quality edges
                if rating >= 4.0:
                    self.edges[HeteroEdges.RESPONSE_HIGH_QUALITY.value].append((response_id, response_id))
                elif rating <= 2.0:
                    self.edges[HeteroEdges.RESPONSE_LOW_QUALITY.value].append((response_id, response_id))

                self.num_responses += 1

                # Store interaction for pairwise training
                self.interactions.append({
                    'user_id': user_id,
                    'session_id': session_id,
                    'query_id': query_id,
                    'llm_id': llm_id,
                    'response_id': response_id,
                    'rating': rating,
                    'question_id': question_id,
                    'turn': turn
                })

        print(f"Graph built: {self.num_users} users, {self.num_sessions} sessions, "
              f"{self.num_queries} queries, {self.num_llms} LLMs, {self.num_responses} responses")
        print(f"Total edge types: {len(self.edges)}")

    def _create_hetero_data(self):
        """
        Create PyTorch Geometric HeteroData object.

        Returns:
            HeteroData: Heterogeneous graph data
        """
        try:
            from torch_geometric.data import HeteroData
        except ImportError:
            print("Warning: PyTorch Geometric not installed. Returning simplified data.")
            return self._create_simplified_data()

        data = HeteroData()

        # Node features
        # User: learned embeddings (initialized as zeros)
        data['user'].x = torch.zeros((self.num_users, self.embedding_dim or 128))

        # Session: learned embeddings (initialized as zeros)
        data['session'].x = torch.zeros((self.num_sessions, self.embedding_dim or 128))

        # Query: pretrained embeddings
        if len(self.query_embeddings) > 0:
            data['query'].x = torch.tensor(np.array(self.query_embeddings), dtype=torch.float32)
        else:
            data['query'].x = torch.zeros((self.num_queries, self.embedding_dim or 128))

        # LLM: pretrained embeddings
        if len(self.llm_embeddings) > 0:
            data['llm'].x = torch.tensor(np.array(self.llm_embeddings), dtype=torch.float32)
        else:
            data['llm'].x = torch.zeros((self.num_llms, self.embedding_dim or 128))

        # Response: rating-based features
        response_features = []
        for interaction in self.interactions:
            rating = interaction['rating']
            # Normalize rating to 0-1 and create feature vector
            normalized_rating = rating / 5.0  # Assuming 0-5 scale
            feature = torch.ones(self.embedding_dim or 128) * normalized_rating
            response_features.append(feature)

        if len(response_features) > 0:
            data['response'].x = torch.stack(response_features)
        else:
            data['response'].x = torch.zeros((self.num_responses, self.embedding_dim or 128))

        # Edge indices
        for edge_type, edge_list in self.edges.items():
            if len(edge_list) == 0:
                continue

            # Parse edge type (e.g., "user_own_session" -> ("user", "own", "session"))
            parts = edge_type.split('_')
            if len(parts) >= 3:
                src_type = parts[0]
                dst_type = parts[-1]
                rel_type = '_'.join(parts[1:-1])
            else:
                continue

            edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
            data[src_type, rel_type, dst_type].edge_index = edge_index

        return data

    def _create_simplified_data(self):
        """Create simplified data structure when PyTorch Geometric is not available."""
        return {
            'num_users': self.num_users,
            'num_sessions': self.num_sessions,
            'num_queries': self.num_queries,
            'num_llms': self.num_llms,
            'num_responses': self.num_responses,
            'user_embeddings': torch.zeros((self.num_users, self.embedding_dim or 128)),
            'session_embeddings': torch.zeros((self.num_sessions, self.embedding_dim or 128)),
            'query_embeddings': torch.tensor(np.array(self.query_embeddings), dtype=torch.float32) if self.query_embeddings else torch.zeros((self.num_queries, self.embedding_dim or 128)),
            'llm_embeddings': torch.tensor(np.array(self.llm_embeddings), dtype=torch.float32) if self.llm_embeddings else torch.zeros((self.num_llms, self.embedding_dim or 128)),
            'edges': self.edges,
            'interactions': self.interactions
        }

    def _create_metadata(self):
        """
        Create metadata for pairwise preference training.

        Returns:
            dict: Metadata with pairwise comparisons
        """
        # Group interactions by (question_id, turn)
        grouped = defaultdict(list)
        for interaction in self.interactions:
            key = (interaction['question_id'], interaction['turn'])
            grouped[key].append(interaction)

        # Create pairwise comparisons
        pairs = []
        for key, group in grouped.items():
            if len(group) < 2:
                continue

            # Create all pairwise combinations
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    inter1 = group[i]
                    inter2 = group[j]

                    # Determine which is better based on rating
                    if inter1['rating'] > inter2['rating']:
                        winner = inter1
                        loser = inter2
                        label = 1  # inter1 is better
                    elif inter2['rating'] > inter1['rating']:
                        winner = inter2
                        loser = inter1
                        label = 0  # inter2 is better
                    else:
                        continue  # Skip ties

                    pairs.append({
                        'winner': winner,
                        'loser': loser,
                        'label': label,
                        'question_id': key[0],
                        'turn': key[1]
                    })

        print(f"Created {len(pairs)} pairwise comparisons for training")

        return {
            'pairs': pairs,
            'user_map': self.user_map,
            'llm_map': self.llm_map,
            'query_map': self.query_map,
            'user_id_to_name': self.user_id_to_name,
            'llm_id_to_name': self.llm_id_to_name
        }
