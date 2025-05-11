import os

from cryptography.hazmat.primitives.ciphers.aead import AESGCM


def generate_key() -> bytes:
    """
    Generate a secure 256-bit AES-GCM key.
    """
    return AESGCM.generate_key(bit_length=256)


def encrypt_data(key: bytes, plaintext: bytes) -> dict:
    """
    Encrypt plaintext with AES-GCM. Returns a dict containing nonce, ciphertext, and tag.
    """
    aesgcm = AESGCM(key)
    nonce = os.urandom(12)
    ciphertext = aesgcm.encrypt(nonce, plaintext, None)
    return {"nonce": nonce.hex(), "ciphertext": ciphertext.hex()}


def decrypt_data(key: bytes, nonce_hex: str, ciphertext_hex: str) -> bytes:
    """
    Decrypt AES-GCM encrypted data. Returns the original plaintext.
    """
    aesgcm = AESGCM(key)
    nonce = bytes.fromhex(nonce_hex)
    ciphertext = bytes.fromhex(ciphertext_hex)
    return aesgcm.decrypt(nonce, ciphertext, None)
