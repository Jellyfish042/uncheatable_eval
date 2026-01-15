import os
import json
import subprocess
import importlib.metadata
from packaging import version
import sys
import re
from typing import Dict, List, Optional


def save_json(my_data, file_name):
    if not os.path.exists("data"):
        os.makedirs("data")

    file_name = file_name.replace(".json", "") + ".json"
    path = os.path.join("data", file_name)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(my_data, f, ensure_ascii=True, indent=4)


def install_requirements(requirements: list[str]):
    """
    Installs or upgrades packages and reloads them if already imported.

    :param requirements: List of packages with potential version specifiers.
    """
    for requirement in requirements:
        package_info = (
            requirement.split("==")
            if "==" in requirement
            else (requirement.split(">=") if ">=" in requirement else (requirement.split("<=") if "<=" in requirement else [requirement]))
        )
        package_name = package_info[0]
        required_version_spec = requirement[len(package_name) :]

        try:
            # Check if the package is already installed
            dist = importlib.metadata.distribution(package_name)
            installed_version = version.parse(dist.version)
            needs_reload = False

            if required_version_spec:
                # Extract the operator and the version from requirement
                operator = required_version_spec[:2] if required_version_spec[1] in ["=", ">"] else required_version_spec[0]
                required_version = version.parse(required_version_spec.lstrip(operator))

                # Version comparison based on the operator
                if (
                    (operator == "==" and installed_version == required_version)
                    or (operator == ">=" and installed_version >= required_version)
                    or (operator == "<=" and installed_version <= required_version)
                ):
                    print(f"Package {package_name} already installed and meets the requirement {requirement}.")
                else:
                    needs_reload = True
                    print(f"Package {package_name} version {installed_version} does not meet the requirement {requirement}, upgrading...")
                    subprocess.check_call([sys.executable, "-m", "pip", "install", f"{package_name}{required_version_spec}"])
            else:
                print(f"Package {package_name} is already installed.")

            if needs_reload:
                # Reload the package if it was already imported
                if package_name in sys.modules:
                    importlib.reload(sys.modules[package_name])
                    print(f"Reloaded {package_name}")
        except importlib.metadata.PackageNotFoundError:
            # Package is not installed, install it with the specified version
            print(f"Package {package_name} is not installed, installing {requirement}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", requirement])
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Error installing or upgrading package {package_name}: {e}")


def bytes_to_unicode() -> Dict[int, str]:
    """
    GPT-2 style byte-to-unicode mapping.
    Maps byte values 0-255 to printable Unicode characters.
    """
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


class TokenizerBytesConverter:
    """
    Universal Token-to-Bytes Converter for HuggingFace tokenizers.

    Supports two encoding schemes:
    1. ByteLevel BPE (Llama 3.x, Qwen, GPT-2 style)
    2. SentencePiece with ByteFallback (Mistral, early LLaMA)

    Usage:
        converter = TokenizerBytesConverter("meta-llama/Llama-3.2-1B")
        nested_bytes = converter.encode_to_bytes("Hello world")
        # Returns: [[72, 101, 108, 108, 111], [32, 119, 111, 114, 108, 100]]
    """

    # Class-level mapping table cache
    _BYTE_TO_UNICODE = bytes_to_unicode()
    _UNICODE_TO_BYTE = {v: k for k, v in _BYTE_TO_UNICODE.items()}

    def __init__(
        self,
        model_name_or_path: str = None,
        cache_dir: Optional[str] = None,
        trust_remote_code: bool = True,
        tokenizer=None,
    ):
        """
        Initialize the converter.

        Args:
            model_name_or_path: HuggingFace model name or local path
            cache_dir: Directory to cache the downloaded tokenizer files
            trust_remote_code: Whether to trust remote code for custom tokenizers
            tokenizer: Optional pre-loaded tokenizer instance for encoding.
                       If provided, this tokenizer will be used for encode() calls,
                       while AutoTokenizer is still used to extract vocab/decoder config.
        """
        from transformers import AutoTokenizer

        # Always load AutoTokenizer for vocab extraction
        auto_tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            cache_dir=cache_dir,
            trust_remote_code=trust_remote_code,
        )

        # Use provided tokenizer for encoding, or fall back to auto_tokenizer
        self._tokenizer = tokenizer if tokenizer is not None else auto_tokenizer

        # Extract tokenizer.json from the AutoTokenizer's backend
        if hasattr(auto_tokenizer, "backend_tokenizer") and hasattr(auto_tokenizer.backend_tokenizer, "to_str"):
            tokenizer_json = json.loads(auto_tokenizer.backend_tokenizer.to_str())
        else:
            raise ValueError("Tokenizer object is not supported. " "The tokenizer must have a backend_tokenizer with to_str() method.")

        self._tokenizer_json = tokenizer_json
        self._vocab = tokenizer_json["model"]["vocab"]
        self._id_to_token: Dict[int, str] = {v: k for k, v in self._vocab.items()}

        # Detect encoding type
        self._decoder_type = self._detect_decoder_type()

        # Load added_tokens
        self._load_added_tokens()

    def _detect_decoder_type(self) -> str:
        """Detect the decoder type from tokenizer.json."""
        decoder = self._tokenizer_json.get("decoder", {})
        decoder_type = decoder.get("type", "")

        if decoder_type == "ByteLevel":
            return "bytelevel"
        elif decoder_type == "Sequence":
            decoders = decoder.get("decoders", [])
            for d in decoders:
                if d.get("type") == "ByteFallback":
                    return "sentencepiece"
            for d in decoders:
                if d.get("type") == "ByteLevel":
                    return "bytelevel"

        # Fallback: check model configuration
        model = self._tokenizer_json.get("model", {})
        if model.get("byte_fallback", False):
            return "sentencepiece"

        # Default to bytelevel
        return "bytelevel"

    def _load_added_tokens(self):
        """Load added_tokens into the vocabulary."""
        added_tokens = self._tokenizer_json.get("added_tokens", [])
        for token_info in added_tokens:
            token_id = token_info["id"]
            content = token_info["content"]
            self._id_to_token[token_id] = content

    @property
    def decoder_type(self) -> str:
        """Return the detected decoder type."""
        return self._decoder_type

    @property
    def vocab_size(self) -> int:
        """Return the vocabulary size."""
        return len(self._id_to_token)

    @property
    def tokenizer(self):
        """Return the underlying HuggingFace tokenizer."""
        return self._tokenizer

    def get_token_string(self, token_id: int) -> Optional[str]:
        """Get the raw string for a token_id."""
        return self._id_to_token.get(token_id)

    def token_to_bytes(self, token_id: int) -> Optional[List[int]]:
        """
        Map a single token_id to its byte sequence.

        Args:
            token_id: The token ID

        Returns:
            List of byte values (0-255) as integers, or None if token_id doesn't exist
        """
        token_str = self._id_to_token.get(token_id)
        if token_str is None:
            return None

        if self._decoder_type == "bytelevel":
            return self._decode_bytelevel(token_str)
        else:
            return self._decode_sentencepiece(token_str)

    def _decode_bytelevel(self, token_str: str) -> List[int]:
        """
        ByteLevel decoding: map each Unicode character back to a byte.
        """
        result = []
        for char in token_str:
            if char in self._UNICODE_TO_BYTE:
                result.append(self._UNICODE_TO_BYTE[char])
            else:
                # Characters not in the mapping table are encoded as UTF-8
                result.extend(char.encode("utf-8"))
        return result

    def _decode_sentencepiece(self, token_str: str) -> List[int]:
        """
        SentencePiece decoding: handle ▁ and <0xXX> format.
        """
        result = []
        i = 0
        while i < len(token_str):
            # Check for <0xXX> format
            match = re.match(r"<0x([0-9A-Fa-f]{2})>", token_str[i:])
            if match:
                byte_val = int(match.group(1), 16)
                result.append(byte_val)
                i += 6
            elif token_str[i] == "▁":
                # Replace ▁ with space
                result.append(0x20)
                i += 1
            else:
                result.extend(token_str[i].encode("utf-8"))
                i += 1
        return result

    def encode_to_bytes(
        self,
        text: str,
        add_special_tokens: bool = False,
        strip_leading_space: bool = True,
    ) -> List[List[int]]:
        """
        Encode text to a nested list of bytes.

        Each sub-list contains the byte values (as integers) for one token.

        Args:
            text: Input text to encode
            add_special_tokens: Whether to add special tokens (BOS, EOS, etc.)
            strip_leading_space: For SentencePiece, whether to strip the leading space
                                from the first token

        Returns:
            Nested list where each inner list contains byte values for one token.
            Example: [[72, 101, 108, 108, 111], [32, 119, 111, 114, 108, 100]]
        """
        token_ids = self._tokenizer.encode(text, add_special_tokens=add_special_tokens)

        result = []
        for idx, token_id in enumerate(token_ids):
            token_bytes = self.token_to_bytes(token_id)
            if token_bytes is not None:
                # Handle SentencePiece leading space
                if idx == 0 and self._decoder_type == "sentencepiece" and strip_leading_space and token_bytes and token_bytes[0] == 0x20:
                    token_bytes = token_bytes[1:]

                result.append(token_bytes)

        return result

    def encode_to_flat_bytes(
        self,
        text: str,
        add_special_tokens: bool = False,
        strip_leading_space: bool = True,
    ) -> bytes:
        """
        Encode text to a flat byte sequence.

        Args:
            text: Input text to encode
            add_special_tokens: Whether to add special tokens
            strip_leading_space: For SentencePiece, whether to strip the leading space

        Returns:
            Concatenated bytes from all tokens
        """
        nested = self.encode_to_bytes(text, add_special_tokens, strip_leading_space)
        result = []
        for token_bytes in nested:
            result.extend(token_bytes)
        return bytes(result)

    def get_all_token_bytes(self) -> Dict[int, List[int]]:
        """
        Get byte mapping for all tokens in the vocabulary.

        Returns:
            Dictionary mapping token_id to list of byte values
        """
        return {token_id: self.token_to_bytes(token_id) for token_id in self._id_to_token}
