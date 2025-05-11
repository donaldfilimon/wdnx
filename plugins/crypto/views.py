import os

from flask import Blueprint, jsonify, request

from plugin_utils.logging import get_logger
from plugin_utils.metrics import metrics
from plugin_utils.validation import require_json_fields

from .encryption_utils import AESGCM, generate_key

crypto_bp = Blueprint("crypto", __name__, url_prefix="/crypto")
logger = get_logger(__name__)


@crypto_bp.route("/generate-key", methods=["GET"])
@metrics
def api_generate_key():
    """Generate a new AES-GCM encryption key (hex-encoded)."""
    key = generate_key().hex()
    logger.info("Generated new AES-GCM key.")
    return jsonify({"key": key})


@crypto_bp.route("/encrypt", methods=["POST"])
@require_json_fields("key", "plaintext")
@metrics
def api_encrypt():
    """Encrypt data using AES-GCM."""
    data = request.get_json(force=True)
    key_hex = data.get("key")
    plaintext = data.get("plaintext")
    logger.debug(f"Attempting to encrypt data with key_hex starting: {key_hex[:4]}...")
    try:
        key = bytes.fromhex(key_hex)
        aes_gcm = AESGCM(key)
        nonce = os.urandom(16)  # Generate a fresh nonce for each encryption
        ciphertext = aes_gcm.encrypt(nonce, plaintext.encode("utf-8"))
        logger.info("Successfully encrypted data.")
        return jsonify({"ciphertext": ciphertext.hex(), "nonce": nonce.hex()})
    except Exception as e:
        logger.error(f"Encryption error: {e}", exc_info=True)
        raise


@crypto_bp.route("/decrypt", methods=["POST"])
@require_json_fields("key", "ciphertext", "nonce")
@metrics
def api_decrypt():
    """Decrypt data using AES-GCM."""
    data = request.get_json(force=True)
    key_hex = data.get("key")
    ciphertext_hex = data.get("ciphertext")
    nonce_hex = data.get("nonce")
    logger.debug(f"Attempting to decrypt data with key_hex starting: {key_hex[:4]}...")
    try:
        key = bytes.fromhex(key_hex)
        ciphertext = bytes.fromhex(ciphertext_hex)
        nonce = bytes.fromhex(nonce_hex)
        aes_gcm = AESGCM(key)
        plaintext = aes_gcm.decrypt(nonce, ciphertext).decode("utf-8")
        logger.info("Successfully decrypted data.")
        return jsonify({"plaintext": plaintext})
    except Exception as e:
        logger.error(f"Decryption error: {e}", exc_info=True)
        raise
