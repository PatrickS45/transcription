import sys
from pathlib import Path

import pytest

# Le projet est à plat à la racine du dépôt (architecture CDC §11).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models import Word  # noqa: E402

FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "C:/Windows/Fonts/arial.ttf",
]


@pytest.fixture(scope="session")
def font_path() -> str:
    for candidate in FONT_CANDIDATES:
        if Path(candidate).is_file():
            return candidate
    pytest.skip("Aucune police TTF disponible pour les tests de rendu")


def make_words(texts: list[str], start: float = 0.0, word_dur: float = 0.3,
               gap: float = 0.05) -> list[Word]:
    """Fabrique une suite de mots horodatés régulière (utilitaire de test)."""
    words = []
    t = start
    for text in texts:
        words.append(Word(text=text, start=t, end=t + word_dur))
        t += word_dur + gap
    return words
