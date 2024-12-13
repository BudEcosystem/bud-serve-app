#  -----------------------------------------------------------------------------
#  Copyright (c) 2024 Bud Ecosystem Inc.
#  #
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#  #
#      http://www.apache.org/licenses/LICENSE-2.0
#  #
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#  -----------------------------------------------------------------------------

"""Provides helper functions for the project."""

from enum import Enum
from typing import Dict, List, Optional, Union

from huggingface_hub.utils import validate_repo_id
from huggingface_hub.utils._validators import HFValidationError


def create_dynamic_enum(enum_name: str, enum_values: List[str]) -> Enum:
    """Create a dynamic Enum class from a list of values.

    This function generates an Enum class with the given name, using the provided values as enum members.
    The enum member names are created by converting the values to uppercase.

    Args:
        enum_name (str): The name of the Enum class to be created.
        enum_values (list): A list of strings representing the values for the Enum members.

    Returns:
        Enum: A dynamically created Enum class with the specified name and members.

    Raises:
        ValueError: If enum_name is not a valid identifier or enum_values is empty.

    Example:
        >>> Color = create_dynamic_enum("Color", ["red", "green", "blue"])
        >>> Color.RED
        <Color.RED: 'red'>
    """
    # creating enum dynamically from a list of values
    # converting enum name to upper assuming no spaces or special characters
    return Enum(enum_name, {val.upper(): val for val in enum_values})


def assign_random_colors_to_names(names: List[str]) -> List[Dict]:
    """Assign random colors to a list of names, trying to avoid color repetition.

    Args:
        names: List of strings to assign colors to

    Returns:
        List of dictionaries containing name and color pairs
        Example: [{"name": "example", "color": "#E57333"}]
    """
    from .constants import DropdownBackgroundColor

    result = []

    for name in names:
        result.append(
            {
                "name": name,
                "color": DropdownBackgroundColor.get_random_color(),
            }
        )

    return result


def normalize_value(value: Optional[Union[str, List, Dict]]) -> Optional[Union[str, List, Dict]]:
    """Normalize a value by handling None, empty strings, empty lists, and empty dicts.

    Args:
        value: The value to normalize

    Returns:
        - None if the value is None, empty string, empty list, or empty dict
        - Stripped string if the value is a non-empty string
        - Original value for non-empty lists and dicts
        - Original value for other types
    """
    if value is None:
        return None

    # Handle strings
    if isinstance(value, str):
        stripped_value = value.strip()
        return stripped_value if stripped_value else None

    # Handle lists
    if isinstance(value, list):
        return value if value else None

    # Handle dicts
    if isinstance(value, dict):
        return value if value else None

    # Return original value for other types
    return value


def validate_huggingface_repo_format(repo_id: str) -> bool:
    """Validate a huggingface repo id.

    Args:
        repo_id: The huggingface repo id to validate

    Returns:
        True if the repo id is valid, False otherwise
    """
    if not isinstance(repo_id, str):
        return False

    repo_id = repo_id.strip()
    if not repo_id:
        return False

    parts = repo_id.split("/")

    if len(parts) != 2:
        return False

    try:
        validate_repo_id(repo_id)
    except HFValidationError:
        return False

    return True
