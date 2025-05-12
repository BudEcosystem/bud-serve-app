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

import os
import random
import string
import re
from pathlib import Path
import shutil
import filecmp
from enum import Enum
from typing import Dict, List, Literal, Optional, Union, Tuple

from huggingface_hub.utils import validate_repo_id
from huggingface_hub.utils._validators import HFValidationError

from budapp.commons import logging


logger = logging.get_logger(__name__)


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


def validate_icon(icon: str) -> bool:
    """Validates if the provided string is either an emoji or a valid path to an icon.

    Args:
        icon (str): String to validate as emoji or path

    Returns:
        bool: True if valid emoji or existing path

    Raises:
        ValueError: If raise_exception is True and icon is invalid
    """
    from .config import app_settings
    from .constants import EMOJIS

    if not icon:
        logger.debug("No icon provided")
        return False

    try:
        if icon in EMOJIS:
            logger.debug(f"Valid emoji icon: {icon}")
            return True

        icon_path = os.path.join(app_settings.static_dir, icon)
        if os.path.exists(icon_path) and os.path.isfile(icon_path):
            logger.debug(f"Valid file icon: {icon}")
            return True

        logger.debug(f"Invalid icon: {icon}")
        return False

    except Exception as e:
        logger.error(f"Error validating icon: {e}")
        return False


def get_hardware_types(cpu_count: int, gpu_count: int, hpu_count: int) -> List[Literal["CPU", "GPU", "HPU"]]:
    """Get list of hardware types based on hardware counts.

    Args:
        cpu_count: Number of CPUs
        gpu_count: Number of GPUs
        hpu_count: Number of HPUs

    Returns:
        List of hardware types available
    """
    hardware = []
    if cpu_count > 0:
        hardware.append("CPU")
    if gpu_count > 0:
        hardware.append("GPU")
    if hpu_count > 0:
        hardware.append("HPU")

    return hardware


def get_param_range(num_params: int) -> tuple[int, int]:
    """Get the parameter range for model comparison based on model size.

    Billion-scale models (B):
        - 200B+ params:     ±100B
        - 100B to 200B:     ±50B
        - 50B to 100B:      ±30B
        - 20B to 50B:       ±10B
        - 10B to 20B:       ±5B
        - 5B to 10B:        ±2B
        - 2B to 5B:         ±1B
        - 1B to 2B:         ±0.5B

    Million-scale models (M):
        - All models ≥1M:   ±100M

    Thousand-scale models (K):
        - All models <1M:   max(±500K, ±50% of params)

    Args:
        num_params: Number of parameters in the model

    Returns:
        tuple[int, int]: (min_params, max_params) as integers
    """
    # Convert to billions
    num_params_in_billions = num_params / 1_000_000_000

    # Handle billion-scale models
    if num_params_in_billions >= 1:
        # Define ranges for billion-scale models
        BILLION_RANGES = [
            (200, 100),  # 200B+ params: ±100B
            (100, 50),  # 100B-200B params: ±50B
            (50, 30),  # 50B-100B params: ±30B
            (20, 10),  # 20B-50B params: ±10B
            (10, 5),  # 10B-20B params: ±5B
            (5, 2),  # 5B-10B params: ±2B
            (2, 1),  # 2B-5B params: ±1B
            (1, 0.5),  # 1B-2B params: ±0.5B
        ]

        for threshold, delta in BILLION_RANGES:
            if num_params_in_billions >= threshold:
                min_params_in_billions = max(0, num_params_in_billions - delta)
                max_params_in_billions = num_params_in_billions + delta

                min_num_params = int(min_params_in_billions * 1_000_000_000)
                max_num_params = int(max_params_in_billions * 1_000_000_000)

                logger.debug(
                    f"Model has {num_params_in_billions:.2f}B params, "
                    f"using +/-{delta:.1f}B range: "
                    f"{min_params_in_billions:.2f}B to {max_params_in_billions:.2f}B"
                )
                return min_num_params, max_num_params

    # Handle million-scale models
    elif num_params >= 1_000_000:
        num_params_in_millions = num_params / 1_000_000
        # Static ±100M range for all million-scale models
        delta_millions = 100

        min_params_in_millions = max(0, num_params_in_millions - delta_millions)
        max_params_in_millions = num_params_in_millions + delta_millions

        min_num_params = int(min_params_in_millions * 1_000_000)
        max_num_params = int(max_params_in_millions * 1_000_000)

        logger.debug(
            f"Model has {num_params_in_millions:.2f}M params, "
            f"using +/-{delta_millions}M range: "
            f"{min_params_in_millions:.2f}M to {max_params_in_millions:.2f}M"
        )
        return min_num_params, max_num_params

    # Handle thousand-scale models
    else:
        num_params_in_thousands = num_params / 1000
        # Use ±500K or half of current value, whichever is larger
        range_size = max(500_000, num_params // 2)

        min_num_params = max(0, num_params - range_size)
        max_num_params = num_params + range_size

        logger.debug(
            f"Model has {num_params_in_thousands:.2f}K params, "
            f"using +/-{range_size/1000:.1f}K range: "
            f"{min_num_params/1000:.2f}K to {max_num_params/1000:.2f}K"
        )
        return min_num_params, max_num_params


def generate_valid_password(min_length: int = 8) -> str:
    """
    Generates a valid password string that meets the following criteria:
    - Contains at least one digit.
    - Contains at least one alphabetic character.
    - Contains at least one special character from the set `!@#$%^&*`.
    - Has a minimum length of `min_length`.

    Args:
        min_length (int, optional): The minimum length of the password. Defaults to 8.

    Returns:
        str: A randomly generated password string that satisfies the above criteria.
    """
    # Define character pools
    digits = string.digits
    alphabets = string.ascii_letters
    special_chars = "!@#$%^&*"

    # Ensure the string contains at least one digit, alphabet, and special character
    mandatory_chars = [
        random.choice(digits),
        random.choice(alphabets),
        random.choice(special_chars),
    ]

    # Combine all allowed characters (excluding whitespaces)
    all_chars = digits + alphabets + special_chars

    # Fill the rest of the string with random characters from the combined pool
    remaining_length = min_length - len(mandatory_chars)
    remaining_chars = [random.choice(all_chars) for _ in range(remaining_length)]

    # Combine and shuffle the characters
    result = mandatory_chars + remaining_chars
    random.shuffle(result)

    # Join the characters into a single string
    final_string = "".join(result)

    return final_string


def validate_password_string(password: str) -> Union[bool, Tuple[bool, str]]:
    """
    Validate the password based on the following conditions:
    - Contains at least one digit
    - Contains at least one alphabet character
    - Contains at least one special character
    - Contains no whitespace

    Args:
        password: The password to validate.

    Returns:
        A tuple with a boolean and an error message if the validation fails,
        or True if the validation succeeds.
    """
    if not re.search(r"\d", password):
        return False, "Password must contain at least one digit."
    if not re.search(r"[a-zA-Z]", password):
        return False, "Password must contain at least one alphabet character."
    if not re.search(f"[{re.escape(string.punctuation)}]", password):
        return False, "Password must contain at least one special character."
    if re.search(r"\s", password):
        return False, "Password must not contain any whitespace."

    return True, "Password is valid."


def replicate_dir(source: str, destination: str, is_override: bool = False):
    """
    Sync files from `source` to `destination` recursively.

    - Preserves directory structure.
    - Skips identical files unless `is_override=True`.
    - Creates necessary directories as needed.
    """
    src_path = Path(source)
    dst_path = Path(destination)

    if not src_path.exists() or not src_path.is_dir():
        logger.warning(f"Source directory does not exist or is not a directory: {source}")
        return

    for src_file in src_path.rglob("*"):  # Recursively go through everything
        if src_file.is_file():
            # Compute the relative path from source base
            rel_path = src_file.relative_to(src_path)

            # Construct full path in destination
            dst_file = dst_path / rel_path

            # Ensure the destination folder exists
            dst_file.parent.mkdir(parents=True, exist_ok=True)

            # Decide whether to copy
            if not dst_file.exists():
                shutil.copy2(src_file, dst_file)
                logger.debug(f"Copied (new): {src_file} -> {dst_file}")
            elif is_override:
                # Only copy if files differ
                if not filecmp.cmp(src_file, dst_file, shallow=False):
                    shutil.copy2(src_file, dst_file)
                    logger.debug(f"Copied (override): {src_file} -> {dst_file}")
                else:
                    logger.debug(f"Skipped (identical): {dst_file}")
            else:
                logger.debug(f"Skipped (exists): {dst_file}")
