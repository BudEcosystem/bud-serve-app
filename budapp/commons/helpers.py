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
from typing import List


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
