#!/usr/bin/env python3
"""
IndicTrans2 Test Script - Simplified Version
=============================================

Fixed to handle model generation properly.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
import sys
from dotenv import load_dotenv

# Load environment variables from .env file
env_file = Path(__file__).parent / ".env"
if env_file.exists():
    load_dotenv(env_file)
    print(f"✓ Loaded .env file from {env_file}")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class IndicTrans2Tester:
    """Test IndicTrans2 on Hindi ghazals from Kaggle dataset."""

    def __init__(self, device: str = "cpu", hf_token: Optional[str] = None):
        """Initialize tester."""
        self.device = device
        self.model = None
        self.tokenizer = None
        self.hf_token = hf_token or os.getenv('HF_TOKEN')

        logger.info(f"Initializing IndicTrans2 Tester (device: {device})")
        self._authenticate_huggingface()
        self._load_model()

    def _authenticate_huggingface(self):
        """Authenticate with HuggingFace using token."""
        try:
            from huggingface_hub import login

            token = self.hf_token

            if not token:
                token_file = Path.home() / ".huggingface" / "token"
                if token_file.exists():
                    token = token_file.read_text().strip()
                    logger.info("✓ Found HuggingFace token from ~/.huggingface/token")

            if not token:
                logger.error("\nMISSING HUGGINGFACE TOKEN")
                logger.error("Create .env file: echo 'HF_TOKEN=hf_YOUR_TOKEN' > .env")
                raise ValueError("HuggingFace token not found")

            logger.info("Authenticating with HuggingFace token...")
            login(token=token, add_to_git_credential=False)
            logger.info("✓ Authentication successful")

        except Exception as e:
            logger.error(f"Authentication failed: {e}")
            raise

    def _load_model(self):
        """Load IndicTrans2 model from HuggingFace."""
        try:
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
            import torch

            logger.info("Loading IndicTrans2 model...")

            model_name = "ai4bharat/indictrans2-indic-en-1B"

            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True,
                use_fast=False
            )

            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                trust_remote_code=True
            )

            if self.device == "cuda":
                self.model = self.model.to(self.device)

            logger.info("✓ Model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def translate_text(self, text: str, source_lang: str = "hin_Deva", target_lang: str = "eng_Latn") -> str:
        """Translate single text with proper error handling."""
        try:
            import torch

            # Format: src_lang tgt_lang text
            input_text = f"{source_lang} {target_lang} {text}"

            # Tokenize
            inputs = self.tokenizer(
                input_text,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )

            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Generate translation - use only valid parameters
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    num_beams=4,
                    max_length=512,
                    num_return_sequences=1
                    # Removed: temperature (not valid for beam search)
                )

            # Check if generation returned valid output
            if generated_ids is None or generated_ids.shape[0] == 0:
                logger.warning("Model generated empty output")
                return ""

            # Decode the output
            translation = self.tokenizer.decode(
                generated_ids[0],
                skip_special_tokens=True
            )

            # Clean up translation
            translation = translation.strip() if translation else ""

            return translation

        except Exception as e:
            logger.error(f"Translation error: {type(e).__name__}: {str(e)[:100]}")
            raise

    def find_dataset(self, base_path: str = ".") -> Optional[Path]:
        """Find dataset folder."""
        logger.info(f"Searching for dataset in: {base_path}")

        base = Path(base_path)

        if not base.exists():
            logger.error(f"Base path not found: {base_path}")
            return None

        patterns = [
            "*dataset*urdu*ghazal*",
            "dataset*",
            "*ghazal*",
        ]

        for pattern in patterns:
            matches = list(base.glob(pattern))
            if matches:
                for match in matches:
                    if match.is_dir():
                        subdirs = list(match.iterdir())
                        if subdirs:
                            first_subdir = subdirs[0]
                            if first_subdir.is_dir():
                                subfolders = [d.name for d in first_subdir.iterdir() if d.is_dir()]
                                if 'en' in subfolders and 'hi' in subfolders and 'ur' in subfolders:
                                    logger.info(f"✓ Found dataset: {match}")
                                    return match

        logger.error("Dataset not found")
        return None

    def load_sample_poems(self, dataset_path: Path, num_samples: int = 2) -> List[Dict]:
        """Load sample Hindi ghazals from dataset."""
        logger.info(f"Loading {num_samples} sample poems from dataset...")

        samples = []

        if not dataset_path.exists():
            logger.error(f"Dataset path not found: {dataset_path}")
            return []

        poets = sorted([d for d in dataset_path.iterdir() if d.is_dir()])

        if not poets:
            logger.error(f"No poet folders found in {dataset_path}")
            return []

        logger.info(f"Found {len(poets)} poets")

        count = 0
        for poet_dir in poets[:5]:
            poet_name = poet_dir.name

            hi_folder = poet_dir / "hi"
            en_folder = poet_dir / "en"

            if not hi_folder.exists() or not en_folder.exists():
                continue

            hi_poems = sorted(list(hi_folder.glob("*")))[:num_samples - count]

            for hi_file in hi_poems:
                if not hi_file.is_file():
                    continue

                try:
                    with open(hi_file, 'r', encoding='utf-8') as f:
                        hindi_text = f.read().strip()
                except Exception as e:
                    logger.warning(f"Could not read {hi_file}: {e}")
                    continue

                if not hindi_text:
                    continue

                en_file = en_folder / hi_file.name
                english_text = ""
                if en_file.exists():
                    try:
                        with open(en_file, 'r', encoding='utf-8') as f:
                            english_text = f.read().strip()
                    except Exception as e:
                        logger.debug(f"Could not read reference {en_file}: {e}")

                samples.append({
                    'poet': poet_name,
                    'poem_id': hi_file.stem,
                    'poem_filename': hi_file.name,
                    'hindi': hindi_text,
                    'english_reference': english_text
                })

                count += 1
                if count >= num_samples:
                    break

            if count >= num_samples:
                break

        logger.info(f"Loaded {len(samples)} samples")
        return samples

    def test_translation(self, dataset_path: Path, num_samples: int = 2) -> List[Dict]:
        """Test translation on sample poems."""
        logger.info(f"\n{'=' * 80}")
        logger.info("TESTING INDICTRANS2 ON HINDI GHAZALS")
        logger.info(f"{'=' * 80}\n")

        samples = self.load_sample_poems(dataset_path, num_samples)

        if not samples:
            logger.error("Could not load any samples")
            return []

        results = []

        for i, sample in enumerate(samples, 1):
            logger.info(f"\n{'─' * 80}")
            logger.info(f"SAMPLE {i}/{len(samples)}")
            logger.info(f"{'─' * 80}")

            poet = sample['poet']
            poem_id = sample['poem_id']
            poem_filename = sample['poem_filename']
            hindi_text = sample['hindi']
            english_ref = sample['english_reference']

            logger.info(f"Poet: {poet}")
            logger.info(f"Poem: {poem_filename}\n")

            hindi_display = hindi_text[:200] + "..." if len(hindi_text) > 200 else hindi_text
            logger.info(f"Hindi Original (Devanagari):\n{hindi_display}\n")

            logger.info("Translating with IndicTrans2...")
            try:
                translation = self.translate_text(hindi_text)
                if translation and translation.strip():
                    logger.info("✓ Translation successful\n")
                else:
                    logger.warning("✗ Translation returned empty result\n")
                    translation = ""
            except Exception as e:
                logger.error(f"✗ Translation failed: {e}\n")
                translation = ""

            trans_display = translation[:200] + "..." if len(translation) > 200 else translation
            logger.info(f"Generated Translation:\n{trans_display if trans_display else '[EMPTY]'}\n")

            if english_ref:
                ref_display = english_ref[:200] + "..." if len(english_ref) > 200 else english_ref
                logger.info(f"Reference English Translation:\n{ref_display}\n")

            hindi_words = len(hindi_text.split())
            trans_words = len(translation.split()) if translation else 0
            ref_words = len(english_ref.split()) if english_ref else 0

            logger.info(f"Statistics:")
            logger.info(f"  Hindi words: {hindi_words}")
            logger.info(f"  Generated words: {trans_words}")
            if english_ref:
                logger.info(f"  Reference words: {ref_words}")
            logger.info(
                f"  Expansion ratio: {trans_words / hindi_words:.2f}x\n" if hindi_words > 0 else "  Expansion ratio: N/A\n")

            results.append({
                'poet': poet,
                'poem_filename': poem_filename,
                'poem_id': poem_id,
                'hindi_original': hindi_text,
                'generated_translation': translation,
                'reference_translation': english_ref,
                'statistics': {
                    'hindi_words': hindi_words,
                    'generated_words': trans_words,
                    'reference_words': ref_words,
                    'expansion_ratio': trans_words / hindi_words if hindi_words > 0 else 0
                }
            })

        logger.info(f"\n{'=' * 80}")
        logger.info("TEST SUMMARY")
        logger.info(f"{'=' * 80}")
        logger.info(f"Total samples tested: {len(results)}")
        logger.info(f"Successful translations: {sum(1 for r in results if r['generated_translation'])}")

        return results

    def save_results(self, results: List[Dict],
                     output_file: str = "../test_results/output/translation_test_results.json"):
        """Save test results to JSON file."""
        logger.info(f"\nSaving results to {output_file}...")

        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        logger.info(f"✓ Results saved to {output_file}")


def main():
    """Main function."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Test IndicTrans2 on Hindi ghazals from Kaggle dataset"
    )

    parser.add_argument(
        '--dataset-path',
        type=str,
        default=None,
        help='Path to dataset folder (auto-finds if not specified)'
    )

    parser.add_argument(
        '--num-samples',
        type=int,
        default=2,
        help='Number of samples to test (default: 2)'
    )

    parser.add_argument(
        '--device',
        type=str,
        choices=['cpu', 'cuda'],
        default='cpu',
        help='Computation device (default: cpu)'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='../test_results/output/translation_test_results.json',
        help='Output file for results'
    )

    parser.add_argument(
        '--hf-token',
        type=str,
        default=None,
        help='HuggingFace API token'
    )

    args = parser.parse_args()

    try:
        tester = IndicTrans2Tester(device=args.device, hf_token=args.hf_token)
    except Exception as e:
        logger.error(f"Failed to initialize: {e}")
        sys.exit(1)

    if args.dataset_path:
        dataset_path = Path(args.dataset_path)
        if not dataset_path.exists():
            logger.error(f"Dataset path not found: {args.dataset_path}")
            sys.exit(1)
    else:
        logger.info("Auto-searching for dataset...")

        search_paths = [".", "./Data/raw", "../Data/raw", "Data/raw"]

        dataset_path = None
        for search_path in search_paths:
            dataset_path = tester.find_dataset(search_path)
            if dataset_path:
                break

        if not dataset_path:
            logger.error("Could not find dataset!")
            sys.exit(1)

    results = tester.test_translation(dataset_path, num_samples=args.num_samples)

    if results:
        tester.save_results(results, args.output)
        logger.info(f"\n✅ Testing complete!")
        logger.info(f"Results saved to: {args.output}")
    else:
        logger.error("No results to save")
        sys.exit(1)


if __name__ == '__main__':
    try:
        import torch
        import transformers
        from huggingface_hub import login
    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        logger.error(
            "Install with: pip install torch transformers huggingface-hub python-dotenv --break-system-packages")
        sys.exit(1)

    main()