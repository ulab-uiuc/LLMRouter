import pickle
import torch
from typing import Any
from pathlib import Path


def save_model(model: Any, filepath: str) -> bool:
    """
    Save a model to either .pt (PyTorch) or .pkl (pickle) file format.
    Automatically detects file type from the file extension.

    Args:
        model: The model object to be saved
        filepath: Full path including filename and extension (.pt or .pkl)

    Returns:
        bool: True if save was successful, False otherwise

    Raises:
        ValueError: If file extension is not supported
        Exception: If the save operation fails
    """
    try:
        # Convert to Path object for easier manipulation
        file_path: Path = Path(filepath)

        # Create directory if it doesn't exist
        directory: Path = file_path.parent
        if not directory.exists():
            directory.mkdir(parents=True, exist_ok=True)
            print(f"Created directory: {directory}")

        # Determine file type from extension
        extension: str = file_path.suffix.lower()

        # Save based on file extension
        if extension == ".pt":
            torch.save(model, filepath)
            print(f"Successfully saved PyTorch model: {filepath}")
        elif extension == ".pkl":
            with open(filepath, 'wb') as file_handle:
                pickle.dump(model, file_handle)
            print(f"Successfully saved pickle model: {filepath}")
        else:
            raise ValueError(f"Unsupported file extension: {extension}. Use .pt or .pkl")

        return True

    except Exception as error:
        print(f"Error saving model to {filepath}: {str(error)}")
        return False


def load_model(filepath: str) -> Any:
    """
    Load a model from either .pt (PyTorch) or .pkl (pickle) file format.
    Automatically detects file type from the file extension.

    Args:
        filepath: Full path including filename and extension (.pt or .pkl)

    Returns:
        Any: The loaded model

    Raises:
        FileNotFoundError: If the file or directory doesn't exist
        ValueError: If file extension is not supported
        Exception: If the load operation fails
    """
    try:
        # Convert to Path object for easier manipulation
        file_path: Path = Path(filepath)

        # Check if directory exists
        directory: Path = file_path.parent
        if not directory.exists():
            raise FileNotFoundError(f"Directory does not exist: {directory}")

        # Check if file exists
        if not file_path.exists():
            raise FileNotFoundError(f"File does not exist: {filepath}")

        # Determine file type from extension
        extension: str = file_path.suffix.lower()

        # Load based on file extension
        loaded_model: Any = None
        if extension == ".pt":
            loaded_model = torch.load(filepath, map_location='cpu')  # Load to CPU by default
            print(f"Successfully loaded PyTorch model: {filepath}")
        elif extension == ".pkl":
            with open(filepath, 'rb') as file_handle:
                loaded_model = pickle.load(file_handle)
            print(f"Successfully loaded pickle model: {filepath}")
        else:
            raise ValueError(f"Unsupported file extension: {extension}. Use .pt or .pkl")

        return loaded_model

    except FileNotFoundError as error:
        print(f"File not found error: {str(error)}")
        raise
    except Exception as error:
        print(f"Error loading model from {filepath}: {str(error)}")
        raise
