"""
Encryption Utilities
Handles state encryption/decryption for secure GitHub backup
"""

import os
import base64
import logging
from typing import Optional, Union
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import json

logger = logging.getLogger(__name__)

class StateEncryption:
    """Handles encryption and decryption of state data"""
    
    def __init__(self):
        self.salt_file = '.salt'
        
    def generate_key_from_passphrase(self, passphrase: str, salt: Optional[bytes] = None) -> tuple:
        """Generate encryption key from passphrase"""
        try:
            # Generate or load salt
            if salt is None:
                if os.path.exists(self.salt_file):
                    with open(self.salt_file, 'rb') as f:
                        salt = f.read()
                else:
                    salt = os.urandom(16)
                    with open(self.salt_file, 'wb') as f:
                        f.write(salt)
            
            # Derive key from passphrase
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100000,
            )
            
            key = base64.urlsafe_b64encode(kdf.derive(passphrase.encode()))
            return key, salt
            
        except Exception as e:
            logger.error(f"Error generating key: {e}")
            raise
    
    def encrypt_data(self, data: Union[dict, str], passphrase: str) -> bytes:
        """Encrypt data with passphrase"""
        try:
            # Convert data to JSON if it's a dict
            if isinstance(data, dict):
                data_str = json.dumps(data, indent=2, default=str)
            else:
                data_str = data
            
            # Generate encryption key
            key, salt = self.generate_key_from_passphrase(passphrase)
            
            # Create cipher
            cipher = Fernet(key)
            
            # Encrypt data
            encrypted_data = cipher.encrypt(data_str.encode())
            
            logger.info("Data encrypted successfully")
            return encrypted_data
            
        except Exception as e:
            logger.error(f"Error encrypting data: {e}")
            raise
    
    def decrypt_data(self, encrypted_data: bytes, passphrase: str) -> dict:
        """Decrypt data with passphrase"""
        try:
            # Generate decryption key
            key, _ = self.generate_key_from_passphrase(passphrase)
            
            # Create cipher
            cipher = Fernet(key)
            
            # Decrypt data
            decrypted_bytes = cipher.decrypt(encrypted_data)
            decrypted_str = decrypted_bytes.decode()
            
            # Parse JSON
            data = json.loads(decrypted_str)
            
            logger.info("Data decrypted successfully")
            return data
            
        except Exception as e:
            logger.error(f"Error decrypting data: {e}")
            raise
    
    def encrypt_file(self, input_file: str, output_file: str, passphrase: str) -> bool:
        """Encrypt a file"""
        try:
            # Read input file
            with open(input_file, 'r') as f:
                data = f.read()
            
            # Encrypt data
            encrypted_data = self.encrypt_data(data, passphrase)
            
            # Write encrypted file
            with open(output_file, 'wb') as f:
                f.write(encrypted_data)
            
            logger.info(f"File encrypted: {input_file} -> {output_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error encrypting file: {e}")
            return False
    
    def decrypt_file(self, input_file: str, output_file: str, passphrase: str) -> bool:
        """Decrypt a file"""
        try:
            # Read encrypted file
            with open(input_file, 'rb') as f:
                encrypted_data = f.read()
            
            # Decrypt data
            data = self.decrypt_data(encrypted_data, passphrase)
            
            # Write decrypted file
            with open(output_file, 'w') as f:
                if isinstance(data, dict):
                    json.dump(data, f, indent=2, default=str)
                else:
                    f.write(str(data))
            
            logger.info(f"File decrypted: {input_file} -> {output_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error decrypting file: {e}")
            return False
    
    def get_passphrase_from_env(self) -> Optional[str]:
        """Get encryption passphrase from environment"""
        passphrase = os.getenv('ENCRYPTION_PASSPHRASE')
        if not passphrase:
            logger.warning("No encryption passphrase found in environment")
            return None
        return passphrase
    
    def prompt_for_passphrase(self, confirm: bool = False) -> str:
        """Prompt user for passphrase (for interactive use)"""
        import getpass
        
        try:
            passphrase = getpass.getpass("Enter encryption passphrase: ")
            
            if confirm:
                confirm_passphrase = getpass.getpass("Confirm passphrase: ")
                if passphrase != confirm_passphrase:
                    raise ValueError("Passphrases do not match")
            
            return passphrase
            
        except KeyboardInterrupt:
            logger.info("Passphrase entry cancelled")
            raise
        except Exception as e:
            logger.error(f"Error getting passphrase: {e}")
            raise
    
    def verify_passphrase(self, passphrase: str, encrypted_data: bytes) -> bool:
        """Verify if passphrase can decrypt the data"""
        try:
            self.decrypt_data(encrypted_data, passphrase)
            return True
        except:
            return False
    
    def change_passphrase(self, old_passphrase: str, new_passphrase: str, encrypted_file: str) -> bool:
        """Change the passphrase for an encrypted file"""
        try:
            # Read encrypted file
            with open(encrypted_file, 'rb') as f:
                encrypted_data = f.read()
            
            # Decrypt with old passphrase
            data = self.decrypt_data(encrypted_data, old_passphrase)
            
            # Encrypt with new passphrase
            new_encrypted_data = self.encrypt_data(data, new_passphrase)
            
            # Write back to file
            with open(encrypted_file, 'wb') as f:
                f.write(new_encrypted_data)
            
            logger.info("Passphrase changed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error changing passphrase: {e}")
            return False

class EncryptionUtils:
    """Utility functions for encryption operations"""
    
    @staticmethod
    def generate_secure_passphrase(length: int = 32) -> str:
        """Generate a cryptographically secure passphrase"""
        import secrets
        import string
        
        alphabet = string.ascii_letters + string.digits + "!@#$%^&*"
        return ''.join(secrets.choice(alphabet) for _ in range(length))
    
    @staticmethod
    def hash_passphrase(passphrase: str) -> str:
        """Hash a passphrase for storage verification"""
        import hashlib
        return hashlib.sha256(passphrase.encode()).hexdigest()
    
    @staticmethod
    def verify_passphrase_strength(passphrase: str) -> dict:
        """Verify passphrase strength"""
        import re
        
        checks = {
            'length': len(passphrase) >= 12,
            'uppercase': bool(re.search(r'[A-Z]', passphrase)),
            'lowercase': bool(re.search(r'[a-z]', passphrase)),
            'digits': bool(re.search(r'\d', passphrase)),
            'special': bool(re.search(r'[!@#$%^&*(),.?":{}|<>]', passphrase))
        }
        
        strength_score = sum(checks.values())
        
        if strength_score >= 4:
            strength = 'strong'
        elif strength_score >= 3:
            strength = 'medium'
        else:
            strength = 'weak'
        
        return {
            'checks': checks,
            'score': strength_score,
            'max_score': 5,
            'strength': strength
        }
