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

"""Provides utility functions for managing security tasks."""

import hashlib

from passlib.context import CryptContext


class HashManager:
    """A class for managing various hashing operations.

    This class provides methods for hashing and verifying passwords using bcrypt,
    as well as creating SHA-256 hashes.

    Attributes:
        pwd_context (CryptContext): A CryptContext instance for password hashing.
    """

    def __init__(self) -> None:
        """Initialize the HashManager with a CryptContext for bcrypt."""
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

    async def get_hash(self, plain_string: str) -> str:
        """Hash a plain string using bcrypt.

        Args:
            plain_string (str): The string to be hashed.

        Returns:
            str: The bcrypt hash of the input string.
        """
        return self.pwd_context.hash(plain_string)

    async def verify_hash(self, plain_string: str, hashed_string: str) -> bool:
        """Verify a plain string against a bcrypt hash.

        Args:
            plain_string (str): The plain string to verify.
            hashed_string (str): The bcrypt hash to verify against.

        Returns:
            bool: True if the plain string matches the hash, False otherwise.
        """
        return self.pwd_context.verify(plain_string, hashed_string)

    @staticmethod
    async def create_sha_256_hash(input_string: str) -> str:
        """Create a SHA-256 hash of the input string.

        Args:
            input_string (str): The string to be hashed.

        Returns:
            str: The hexadecimal representation of the SHA-256 hash.
        """
        # Convert the input string to bytes
        input_bytes = input_string.encode("utf-8")

        # Create a SHA-256 hash object
        sha_256_hash = hashlib.sha256(input_bytes)

        # Get the hexadecimal representation of the hash
        return sha_256_hash.hexdigest()
