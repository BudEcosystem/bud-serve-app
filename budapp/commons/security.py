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
