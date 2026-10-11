from __future__ import annotations

import csv
import os
import posixpath
import re
import sys
import textwrap
import unicodedata
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from xml.etree import ElementTree as ET

# Optional, lightweight English stemmer. PorterStemmer needs no downloaded corpus.
try:
    import nltk
    from nltk.stem import PorterStemmer

    ENGLISH_STEMMER = PorterStemmer()
    NLTK_VERSION = nltk.__version__
except ImportError:
    ENGLISH_STEMMER = None
    NLTK_VERSION = "Not installed"

BASE_DIR = Path(__file__).resolve().parent
DATASET_BASENAME = "NLP_Three_Languages_Dataset"
LANGUAGES = ("English", "Hindi", "Marathi")
EXPECTED_SENTENCE_ROWS = 50

# Conservative demonstration lists for this lab dataset. These are not intended
# to be complete linguistic stop-word dictionaries. Negation words are retained
# because removing them can reverse the meaning of a sentence.
STOP_WORDS = {
    "English": set("""
        a an the and or but if while as at by for from in into of on onto to up with
        is am are was were be been being do does did have has had this that these those
        it its we you they he she i me my our your their his her them us than then very
        can could should would will just about over under again further once
    """.split()),
    "Hindi": set("""
        और या तथा लेकिन क्योंकि कि का की के को में से पर तक लिए द्वारा साथ यह वह ये वे
        इस उस इन उन एक अनेक कुछ कोई सभी हर बहुत भी ही तो जब तब जो जैसा जैसे कैसे कहाँ
        कौन क्या क्यों हम मैं मुझे मेरा मेरी मेरे आप आपका आपकी अपने अपनी हैं है था थी थे
        हो होगा होगी होंगे करता करती करते करना करने किया किये कीजिए द्वारा परंतु अथवा
    """.split()),
    "Marathi": set("""
        आणि किंवा पण तसे कारण की चा ची चे च्या ला ली ले मध्ये पासून साठी कडून वर खाली
        हा ही हे तो ती ते या त्या एक काही कोण सर्व प्रत्येक खूप फार आहे आहेत होता होती
        होते होईल मी मला माझा माझी माझे आम्ही आपण तुम्ही तुमचा तुमची त्याचा त्याची
        त्याचे त्यांच्या त्यांना म्हणून करून करतो करते करतात झाले झाली झालेले जे जसे
        साठीचा याचा याची याचे तसेच म्हणजे
    """.split()),
}

# Sentence punctuation used in English and Devanagari. The sample has one aligned
# sentence in each cell, but this also supports cells containing multiple sentences.
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?।॥])\s+|[\r\n]+")
WHITESPACE_RE = re.compile(r"\s+")
SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([,.;:!?।॥])")


# Professional terminal styling. Colours are shown in interactive terminals,
# including the VS Code integrated terminal, and disabled for redirected output.
COLOUR_ENABLED = sys.stdout.isatty() and "NO_COLOR" not in os.environ

_COLOURS = {
    "blue": "\033[94m",
    "cyan": "\033[96m",
    "green": "\033[92m",
    "yellow": "\033[93m",
    "magenta": "\033[95m",
    "bold": "\033[1m",
    "muted": "\033[90m",
    "reset": "\033[0m",
}


def colour(text: Any, *styles: str) -> str:
    """Apply ANSI styling only when supported by the current terminal."""
    if not COLOUR_ENABLED:
        return str(text)
    prefix = "".join(_COLOURS.get(style, "") for style in styles)
    return f"{prefix}{text}{_COLOURS['reset']}"


def print_rule(char: str = "─", style: str = "cyan", width: int = 88) -> None:
    print(colour(char * width, style))


def print_banner(title: str, subtitle: Optional[str] = None) -> None:
    print()
    print_rule("═", "cyan", 88)
    print(colour(title.center(88), "bold", "cyan"))
    if subtitle:
        print(colour(subtitle.center(88), "muted"))
    print_rule("═", "cyan", 88)


def print_section(title: str, style: str = "blue") -> None:
    print()
    print(colour(f"▸ {title}", "bold", style))
    print(colour("─" * min(88, max(28, len(title) + 4)), "muted"))


def print_status(label: str, value: Any) -> None:
    print(f"  {colour(label + ':', 'muted')} {colour(value, 'green')}")


def normalize_text(value: Any) -> str:
    """Apply Unicode NFC normalization, lowercase/casefold, and whitespace cleanup."""
    text = "" if value is None else str(value)
    text = unicodedata.normalize("NFC", text)
    # Casefold affects Latin text; Devanagari text is unchanged by this operation.
    text = text.casefold()
    text = text.replace("\u200b", "").replace("\ufeff", "")
    text = WHITESPACE_RE.sub(" ", text).strip()
    text = SPACE_BEFORE_PUNCT_RE.sub(r"\1", text)
    return text


def split_sentences(text: str) -> List[str]:
    """Split normalized text into sentences using language-appropriate punctuation."""
    return [part.strip() for part in SENTENCE_SPLIT_RE.split(text) if part.strip()]


def tokenize(text: str) -> List[str]:
    """Tokenize Unicode text and preserve Devanagari combining marks.

    Letters, numbers, and combining marks remain in each token. Punctuation and
    whitespace separate tokens. This avoids splitting Hindi/Marathi vowel signs.
    """
    tokens: List[str] = []
    current: List[str] = []

    for char in text:
        category = unicodedata.category(char)
        if category.startswith("L") or category.startswith("N"):
            current.append(char)
        elif category.startswith("M") and current:
            current.append(char)
        else:
            if current:
                tokens.append("".join(current))
                current = []

    if current:
        tokens.append("".join(current))
    return tokens


def remove_stop_words(tokens: Sequence[str], language: str) -> List[str]:
    stop_set = STOP_WORDS[language]
    return [token for token in tokens if token.casefold() not in stop_set]


def stem_tokens(tokens: Sequence[str], language: str) -> Tuple[List[str], str]:
    """Apply only a supported stemmer; never claim unsupported stemming was done."""
    if language == "English" and ENGLISH_STEMMER is not None:
        return [ENGLISH_STEMMER.stem(token) for token in tokens], "NLTK PorterStemmer applied"
    if language == "English":
        return [], "Not applied: install NLTK to enable Porter stemming"
    return [], "Not applied: no Hindi/Marathi stemmer configured in this script"


def _column_index(cell_reference: str) -> int:
    """Convert an Excel column reference (e.g. A, AB) to zero-based index."""
    letters = re.match(r"[A-Z]+", cell_reference.upper())
    if not letters:
        raise ValueError(f"Invalid Excel cell reference: {cell_reference}")
    result = 0
    for char in letters.group(0):
        result = result * 26 + (ord(char) - ord("A") + 1)
    return result - 1


def _all_text(element: ET.Element) -> str:
    """Concatenate text nodes, including rich-text runs in an XLSX shared string."""
    return "".join(node.text or "" for node in element.iter() if node.tag.endswith("}t"))


def read_xlsx(path: Path) -> List[Dict[str, Any]]:
    """Read the first worksheet in .xlsx using Python's standard library only."""
    main_ns = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    office_rel_ns = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
    package_rel_ns = "{http://schemas.openxmlformats.org/package/2006/relationships}"

    try:
        with zipfile.ZipFile(path, "r") as archive:
            names = set(archive.namelist())
            workbook_root = ET.fromstring(archive.read("xl/workbook.xml"))
            sheets_node = workbook_root.find(f"{main_ns}sheets")
            if sheets_node is None or not list(sheets_node):
                raise ValueError("The XLSX workbook does not contain a worksheet.")
            first_sheet = list(sheets_node)[0]
            relationship_id = first_sheet.attrib.get(f"{office_rel_ns}id")
            if not relationship_id:
                raise ValueError("Could not identify the first worksheet in the XLSX file.")

            rels_root = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
            target = None
            for relationship in rels_root.findall(f"{package_rel_ns}Relationship"):
                if relationship.attrib.get("Id") == relationship_id:
                    target = relationship.attrib.get("Target")
                    break
            if not target:
                raise ValueError("Could not find the worksheet file inside the XLSX workbook.")

            if target.startswith("/"):
                sheet_path = target.lstrip("/")
            else:
                sheet_path = posixpath.normpath(posixpath.join("xl", target))
            if sheet_path not in names:
                raise ValueError(f"Worksheet XML not found inside workbook: {sheet_path}")

            shared_strings: List[str] = []
            if "xl/sharedStrings.xml" in names:
                shared_root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
                shared_strings = [_all_text(item) for item in shared_root.findall(f"{main_ns}si")]

            sheet_root = ET.fromstring(archive.read(sheet_path))
            sheet_data = sheet_root.find(f"{main_ns}sheetData")
            if sheet_data is None:
                raise ValueError("The first worksheet is empty.")

            matrix: List[List[Any]] = []
            for row_element in sheet_data.findall(f"{main_ns}row"):
                cells: Dict[int, Any] = {}
                for cell in row_element.findall(f"{main_ns}c"):
                    ref = cell.attrib.get("r", "")
                    if not ref:
                        continue
                    col = _column_index(ref)
                    cell_type = cell.attrib.get("t", "")
                    value_element = cell.find(f"{main_ns}v")

                    if cell_type == "inlineStr":
                        value: Any = _all_text(cell)
                    elif value_element is None or value_element.text is None:
                        value = ""
                    elif cell_type == "s":
                        index = int(value_element.text)
                        value = shared_strings[index] if index < len(shared_strings) else ""
                    elif cell_type == "b":
                        value = value_element.text == "1"
                    else:
                        raw = value_element.text
                        # IDs like 1.0 should be exported/read as "1", not "1.0".
                        try:
                            number = float(raw)
                            value = int(number) if number.is_integer() else number
                        except (TypeError, ValueError):
                            value = raw
                    cells[col] = value

                if cells:
                    width = max(cells) + 1
                    matrix.append([cells.get(i, "") for i in range(width)])

    except zipfile.BadZipFile as error:
        raise ValueError(f"The file is not a valid .xlsx workbook: {path.name}") from error

    if not matrix:
        raise ValueError("No rows were found in the first worksheet.")

    headers = [str(value).strip() for value in matrix[0]]
    records: List[Dict[str, Any]] = []
    for row in matrix[1:]:
        if not any(str(value).strip() for value in row):
            continue
        padded = row + [""] * max(0, len(headers) - len(row))
        records.append({headers[i]: padded[i] for i in range(len(headers))})
    return records


def read_csv(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames:
            raise ValueError("The CSV file has no header row.")
        return [row for row in reader if any(str(value or "").strip() for value in row.values())]


def locate_dataset() -> Path:
    """Prefer the teacher's CSV when present; accept XLSX as well."""
    candidates = [
        BASE_DIR / f"{DATASET_BASENAME}.csv",
        BASE_DIR / f"{DATASET_BASENAME}.xlsx",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    # Permit a slightly different filename, while still checking file extensions.
    for extension in (".csv", ".xlsx"):
        matches = sorted(
            file for file in BASE_DIR.glob(f"*{extension}")
            if DATASET_BASENAME.casefold() in file.stem.casefold()
        )
        if matches:
            return matches[0]

    expected = " or ".join(path.name for path in candidates)
    raise FileNotFoundError(
        f"Dataset not found. Put {expected} in the same folder as this Python script."
    )


def get_required_columns(records: List[Dict[str, Any]]) -> Dict[str, str]:
    if not records:
        raise ValueError("The dataset has a header but contains no data rows.")
    actual_headers = list(records[0].keys())
    mapping = {str(name).strip().casefold(): str(name) for name in actual_headers}
    required = ("english", "hindi", "marathi")
    missing = [name.title() for name in required if name not in mapping]
    if missing:
        raise ValueError(
            "Missing required column(s): " + ", ".join(missing)
            + ". Expected columns are ID, English, Hindi, Marathi."
            + f" Found: {actual_headers}"
        )
    return mapping


def display_id(value: Any, fallback: int) -> str:
    if value is None or not str(value).strip():
        return str(fallback)
    text = str(value).strip()
    try:
        number = float(text)
        if number.is_integer():
            return str(int(number))
    except ValueError:
        pass
    return text


def write_csv(path: Path, headers: Sequence[str], rows: Sequence[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(headers), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)



def tokenize_with_punctuation(text: str) -> List[str]:
    """Return word tokens plus standalone punctuation/symbol tokens for demonstration."""
    tokens: List[str] = []
    current: List[str] = []

    def flush_current() -> None:
        if current:
            tokens.append("".join(current))
            current.clear()

    for char in text:
        category = unicodedata.category(char)
        if category.startswith("L") or category.startswith("N"):
            current.append(char)
        elif category.startswith("M") and current:
            # Preserve Devanagari vowel signs, nukta and other combining marks.
            current.append(char)
        else:
            flush_current()
            if not char.isspace() and category[:1] in {"P", "S"}:
                tokens.append(char)

    flush_current()
    return tokens


def print_wrapped(label: str, value: str, width: int = 108) -> None:
    """Print a long sample across readable terminal lines."""
    print(label)
    if not value:
        print("   [empty]")
        return
    for line in textwrap.wrap(
        value,
        width=width,
        break_long_words=False,
        break_on_hyphens=False,
        replace_whitespace=True,
    ):
        print("   " + line)


def print_list_wrapped(label: str, values: Sequence[str], limit: Optional[int] = None) -> None:
    shown = list(values if limit is None else values[:limit])
    suffix = ""
    if limit is not None and len(values) > limit:
        suffix = f" ... [showing first {limit} of {len(values)}]"
    print_wrapped(label, " | ".join(shown) + suffix)


def print_detailed_walkthrough(
    source_rows: List[Dict[str, Any]],
    columns: Dict[str, str],
    sample_rows: int = 7,
) -> None:
    """Display a step-by-step walkthrough using the first real rows from the dataset."""
    print_banner(
        f"DETAILED WALKTHROUGH | FIRST {sample_rows} DATASET ROWS",
        "Language-wise NLP processing"
    )

    id_column = columns.get("id")

    for language in LANGUAGES:
        selected_rows: List[Tuple[str, str]] = []
        for row_number, row in enumerate(source_rows, start=1):
            text = str(row.get(columns[language.casefold()]) or "").strip()
            if not text:
                continue
            row_id = display_id(row.get(id_column) if id_column else None, row_number)
            selected_rows.append((row_id, text))
            if len(selected_rows) >= sample_rows:
                break

        original_lines = [text for _, text in selected_rows]
        normalized_lines = [normalize_text(text) for text in original_lines]
        normalized_sample = "\n".join(normalized_lines)

        print()
        print_rule("═", "magenta", 88)
        print(colour(f"  {language.upper()} — REPRESENTATIVE SAMPLES", "bold", "magenta"))
        print_rule("═", "magenta", 88)

        print("\n" + colour(f"[1] RAW TEXT — FIRST {len(selected_rows)} NON-EMPTY ROWS", "bold", "cyan"))
        for row_id, text in selected_rows:
            print_wrapped(f"   Dataset ID {row_id}:", text, width=103)

        print("\n" + colour("[2] NORMALIZED TEXT", "bold", "cyan"))
        for (row_id, _), normalized in zip(selected_rows, normalized_lines):
            print_wrapped(f"   Dataset ID {row_id}:", normalized, width=103)
        changed = any(raw != norm for raw, norm in zip(original_lines, normalized_lines))
        print(f"   Normalization changed at least one sample: {changed}")

        print("\n" + colour("[3] SENTENCE SEGMENTATION", "bold", "cyan"))
        program_sentences: List[str] = []
        for line in normalized_lines:
            program_sentences.extend(split_sentences(line))
        print(f"   Input dataset rows: {len(selected_rows)}")
        print(f"   Sentences found in these rows: {len(program_sentences)}")
        for index, sentence in enumerate(program_sentences, start=1):
            print_wrapped(f"   S{index}:", sentence, width=103)

        print("\n" + colour("[4] TOKENS INCLUDING PUNCTUATION", "bold", "cyan"))
        punct_tokens = tokenize_with_punctuation(normalized_sample)
        print(f"   Token count: {len(punct_tokens)}")
        print_list_wrapped("   Tokens:", punct_tokens, limit=100)

        print("\n" + colour("[5] WORDS AFTER PUNCTUATION REMOVAL", "bold", "cyan"))
        words = tokenize(normalized_sample)
        print(f"   Word tokens: {len(words)}")
        print(f"   Unique word tokens: {len(set(words))}")
        print_list_wrapped("   Words:", words, limit=100)

        print("\n" + colour("[6] STOP-WORD REMOVAL", "bold", "cyan"))
        stop_set = STOP_WORDS[language]
        removed_words = [word for word in words if word.casefold() in stop_set]
        content_words = remove_stop_words(words, language)
        print(f"   Stop words removed ({len(removed_words)} occurrences):")
        print_list_wrapped("   Removed:", removed_words, limit=100)
        print(f"   Content words remaining: {len(content_words)}")
        print_list_wrapped("   Content words:", content_words, limit=100)

        print("\n" + colour("[7] STEMMING / LEMMATIZATION STATUS (FIRST 15 CONTENT WORDS)", "bold", "cyan"))
        if language == "English" and ENGLISH_STEMMER is not None:
            preview = [
                f"{word} -> {ENGLISH_STEMMER.stem(word)}"
                for word in content_words[:15]
            ]
            print_list_wrapped("   NLTK PorterStemmer:", preview, limit=15)
            print("   Lemmatization: not applied (no lemmatizer configured).")
        elif language == "English":
            print("   English stemming skipped because NLTK is not installed.")
            print("   Install using: python -m pip install nltk")
        else:
            print("   Stemming/lemmatization not applied: no validated language-specific")
            print("   stemmer/lemmatizer is configured in this script.")

        print("\n" + colour("[8] ADDITIONAL NLP TASK: WORD-FREQUENCY ANALYSIS", "bold", "cyan"))
        sample_frequencies = Counter(content_words)
        print(f"   Top content words: {sample_frequencies.most_common(10)}")

    print()
    print_rule("═", "cyan", 88)
    print(colour("REPRESENTATIVE WALKTHROUGH COMPLETE", "bold", "green"))
    print_rule("═", "cyan", 88)


def run() -> None:
    input_path = locate_dataset()
    if input_path.suffix.casefold() == ".csv":
        source_rows = read_csv(input_path)
    elif input_path.suffix.casefold() == ".xlsx":
        source_rows = read_xlsx(input_path)
    else:
        raise ValueError("Supported dataset formats are .csv and .xlsx")

    columns = get_required_columns(source_rows)
    id_column = columns.get("id")

    processed_rows: List[Dict[str, Any]] = []
    word_counts: Dict[str, Counter] = {language: Counter() for language in LANGUAGES}
    all_word_counts: Dict[str, Counter] = {language: Counter() for language in LANGUAGES}
    stem_counts: Dict[str, Counter] = {language: Counter() for language in LANGUAGES}
    stats: Dict[str, Dict[str, Any]] = {
        language: {
            "non_empty_cells": 0,
            "raw_characters": 0,
            "sentences": 0,
            "tokens_including_punctuation": 0,
            "tokens_before_stopword_removal": 0,
            "tokens_after_stopword_removal": 0,
            "stopwords_removed": 0,
            "unique_tokens_before_stopword_removal": 0,
            "unique_tokens_after_stopword_removal": 0,
            "stemming_applied_tokens": 0,
            "unique_stems": 0,
        }
        for language in LANGUAGES
    }

    for row_number, row in enumerate(source_rows, start=1):
        row_id = display_id(row.get(id_column) if id_column else None, row_number)

        for language in LANGUAGES:
            column_name = columns[language.casefold()]
            original = str(row.get(column_name) or "").strip()
            if not original:
                continue

            stats[language]["non_empty_cells"] += 1
            stats[language]["raw_characters"] += len(original)
            normalized_full = normalize_text(original)
            sentences = split_sentences(normalized_full)
            if not sentences and normalized_full:
                sentences = [normalized_full]

            for sentence_number, sentence in enumerate(sentences, start=1):
                tokens_with_punctuation = tokenize_with_punctuation(sentence)
                tokens = tokenize(sentence)
                filtered_tokens = remove_stop_words(tokens, language)
                removed_tokens = [token for token in tokens if token.casefold() in STOP_WORDS[language]]
                stemmed_tokens, stemming_status = stem_tokens(filtered_tokens, language)

                all_word_counts[language].update(tokens)
                word_counts[language].update(filtered_tokens)
                if stemmed_tokens:
                    stem_counts[language].update(stemmed_tokens)

                stats[language]["sentences"] += 1
                stats[language]["tokens_including_punctuation"] += len(tokens_with_punctuation)
                stats[language]["tokens_before_stopword_removal"] += len(tokens)
                stats[language]["tokens_after_stopword_removal"] += len(filtered_tokens)
                stats[language]["stopwords_removed"] += len(removed_tokens)
                stats[language]["stemming_applied_tokens"] += len(stemmed_tokens)

                processed_rows.append({
                    "ID": row_id,
                    "Language": language,
                    "Sentence_No": sentence_number,
                    "Original_Text": original,
                    "Normalized_Text": sentence,
                    "Tokens_With_Punctuation": " | ".join(tokens_with_punctuation),
                    "Tokens_Before_Stopword_Removal": " | ".join(tokens),
                    "Stopwords_Removed": " | ".join(removed_tokens),
                    "Tokens_After_Stopword_Removal": " | ".join(filtered_tokens),
                    "Stemmed_Tokens": " | ".join(stemmed_tokens),
                    "Stopwords_Removed_Count": len(removed_tokens),
                    "Stemming_Status": stemming_status,
                })

    for language in LANGUAGES:
        stats[language]["unique_tokens_before_stopword_removal"] = len(all_word_counts[language])
        stats[language]["unique_tokens_after_stopword_removal"] = len(word_counts[language])
        stats[language]["unique_stems"] = len(stem_counts[language])

    processed_path = BASE_DIR / "NLP_Processed_Output.csv"
    frequencies_path = BASE_DIR / "NLP_Word_Frequencies.csv"
    comparison_path = BASE_DIR / "NLP_Language_Comparison.csv"
    summary_path = BASE_DIR / "NLP_Assignment_Summary.txt"

    processed_headers = [
        "ID", "Language", "Sentence_No", "Original_Text", "Normalized_Text",
        "Tokens_With_Punctuation", "Tokens_Before_Stopword_Removal", "Stopwords_Removed",
        "Tokens_After_Stopword_Removal", "Stemmed_Tokens", "Stopwords_Removed_Count",
        "Stemming_Status",
    ]
    write_csv(processed_path, processed_headers, processed_rows)

    frequency_rows: List[Dict[str, Any]] = []
    for language in LANGUAGES:
        for rank, (word, count) in enumerate(word_counts[language].most_common(), start=1):
            frequency_rows.append({
                "Language": language,
                "Rank": rank,
                "Word": word,
                "Frequency": count,
            })
    write_csv(frequencies_path, ["Language", "Rank", "Word", "Frequency"], frequency_rows)

    comparison_rows: List[Dict[str, Any]] = []
    for language in LANGUAGES:
        item = stats[language]
        average_tokens = (
            item["tokens_before_stopword_removal"] / item["sentences"]
            if item["sentences"] else 0
        )
        ttr = (
            item["unique_tokens_before_stopword_removal"] / item["tokens_before_stopword_removal"]
            if item["tokens_before_stopword_removal"] else 0
        )
        if language == "English" and ENGLISH_STEMMER is not None:
            stemming_method = "NLTK PorterStemmer"
        elif language == "English":
            stemming_method = "Not applied (install NLTK)"
        else:
            stemming_method = "Not applied (no language-specific stemmer configured)"
        comparison_rows.append({
            "Language": language,
            "Raw_Characters": item["raw_characters"],
            "Non_Empty_Source_Cells": item["non_empty_cells"],
            "Sentences_After_Segmentation": item["sentences"],
            "Tokens_Including_Punctuation": item["tokens_including_punctuation"],
            "Tokens_Before_Stopword_Removal": item["tokens_before_stopword_removal"],
            "Unique_Tokens_Before_Stopword_Removal": item["unique_tokens_before_stopword_removal"],
            "Type_Token_Ratio_Before_Stopword_Removal": f"{ttr:.3f}",
            "Tokens_After_Stopword_Removal": item["tokens_after_stopword_removal"],
            "Stopwords_Removed": item["stopwords_removed"],
            "Stop_Word_List_Size": len(STOP_WORDS[language]),
            "Unique_Tokens_After_Stopword_Removal": item["unique_tokens_after_stopword_removal"],
            "Average_Tokens_Per_Sentence": f"{average_tokens:.2f}",
            "Stemming_Method": stemming_method,
            "Stemming_Applied_Tokens": item["stemming_applied_tokens"],
            "Unique_Stems": item["unique_stems"],
        })
    write_csv(comparison_path, list(comparison_rows[0].keys()), comparison_rows)

    summary_lines = [
        "ASSIGNMENT 2: FUNDAMENTAL LANGUAGE PROCESSING FOR ENGLISH, HINDI AND MARATHI",
        "=" * 84,
        f"Input dataset: {input_path.name}",
        f"Data rows loaded: {len(source_rows)}",
        f"Non-empty language text cells processed: {sum(stats[language]['non_empty_cells'] for language in LANGUAGES)}",
        f"Python version: {sys.version.split()[0]}",
        f"NLTK version: {NLTK_VERSION}",
        "",
        "OPERATIONS IMPLEMENTED",
        "1. Unicode NFC normalization, case normalization, and whitespace cleanup.",
        "2. Rule-based sentence segmentation using English and Devanagari punctuation.",
        "3. Unicode-aware tokenization; combining marks in Devanagari text are preserved.",
        "4. Language-specific demonstration stop-word removal for English, Hindi, and Marathi.",
        "5. English Porter stemming when NLTK is installed; no Hindi/Marathi stemmer is claimed.",
        "6. Word-frequency analysis and language-wise comparison.",
        "",
        "FULL-DATASET LANGUAGE-WISE RESULTS",
        "-" * 84,
    ]

    for language in LANGUAGES:
        item = stats[language]
        average_tokens = (
            item["tokens_before_stopword_removal"] / item["sentences"]
            if item["sentences"] else 0
        )
        ttr = (
            item["unique_tokens_before_stopword_removal"] / item["tokens_before_stopword_removal"]
            if item["tokens_before_stopword_removal"] else 0
        )
        summary_lines.extend([
            "",
            language,
            f"  Raw characters: {item['raw_characters']}",
            f"  Non-empty source cells: {item['non_empty_cells']}",
            f"  Sentences after segmentation: {item['sentences']}",
            f"  Tokens including punctuation: {item['tokens_including_punctuation']}",
            f"  Word tokens before stop-word removal: {item['tokens_before_stopword_removal']}",
            f"  Unique word tokens before stop-word removal: {item['unique_tokens_before_stopword_removal']}",
            f"  Type-token ratio before stop-word removal: {ttr:.3f}",
            f"  Tokens after stop-word removal: {item['tokens_after_stopword_removal']}",
            f"  Stop words removed: {item['stopwords_removed']}",
            f"  Stop-word list size: {len(STOP_WORDS[language])}",
            f"  Unique tokens after stop-word removal: {item['unique_tokens_after_stopword_removal']}",
            f"  Average tokens per sentence: {average_tokens:.2f}",
            f"  Stemming method: {comparison_rows[LANGUAGES.index(language)]['Stemming_Method']}",
            f"  Stemming applied token count: {item['stemming_applied_tokens']}",
            f"  Unique stems: {item['unique_stems']}",
            "  Top 10 words after stop-word removal:",
        ])
        top_words = word_counts[language].most_common(10)
        if top_words:
            summary_lines.extend([f"    {word}: {count}" for word, count in top_words])
        else:
            summary_lines.append("    No tokens available")

    summary_lines.extend([
        "",
        "COMPARATIVE SUMMARY",
        "-" * 84,
        f"{'Parameter':<43} {'English':>12} {'Hindi':>12} {'Marathi':>12}",
        "-" * 84,
    ])
    comparison_specs = [
        ("Raw characters", "raw_characters"),
        ("Sentences processed", "sentences"),
        ("Tokens including punctuation", "tokens_including_punctuation"),
        ("Word tokens before stop-word removal", "tokens_before_stopword_removal"),
        ("Unique tokens before stop-word removal", "unique_tokens_before_stopword_removal"),
        ("Tokens after stop-word removal", "tokens_after_stopword_removal"),
        ("Stop words removed", "stopwords_removed"),
        ("Unique content tokens", "unique_tokens_after_stopword_removal"),
        ("Stop-word list size", None),
        ("Stemming-applied tokens", "stemming_applied_tokens"),
        ("Unique stems", "unique_stems"),
    ]
    for label, key in comparison_specs:
        values = []
        for language in LANGUAGES:
            value = len(STOP_WORDS[language]) if key is None else stats[language][key]
            values.append(str(value))
        summary_lines.append(f"{label:<43} {values[0]:>12} {values[1]:>12} {values[2]:>12}")

    summary_lines.extend([
        "",
        "ANALYSIS AND LIMITATIONS",
        "- Word-frequency analysis is the selected additional NLP task.",
        "- Word frequencies are calculated after stop-word removal.",
        "- The stop-word lists are small, manually defined demonstration lists, not complete linguistic resources.",
        "- The number of removed stop words depends on these lists and should not be interpreted as an inherent property of a language.",
        "- The sentence splitter is rule-based and may not handle every abbreviation or language-specific edge case.",
        "- English stemming uses NLTK PorterStemmer only when NLTK is installed.",
        "- Hindi and Marathi stemming/lemmatization has not been applied because no validated stemmer is configured.",
        "- POS tagging and named-entity recognition are not implemented; frequency analysis is the chosen additional task.",
        "",
        "OUTPUT FILES",
        f"  {processed_path.name}: sentence-level text, tokenization, removed stop words and stemming status.",
        f"  {frequencies_path.name}: ranked word frequencies for all three languages.",
        f"  {comparison_path.name}: detailed full-dataset comparative metrics.",
        f"  {summary_path.name}: this text report.",
    ])
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print_banner(
        "MULTILINGUAL NLP ASSIGNMENT — RESULTS",
        "Fundamental Language Processing | English • Hindi • Marathi"
    )
    print_status("Dataset", input_path.name)
    print_status("Data rows loaded", len(source_rows))
    print_status("Text samples processed", sum(stats[language]["non_empty_cells"] for language in LANGUAGES))
    print_status("Python version", sys.version.split()[0])
    print_status("NLTK version", NLTK_VERSION)

    print_section("FULL-DATASET COMPARATIVE SUMMARY", "cyan")
    print(
        colour(f"{'Metric':<42}", "bold", "yellow")
        + colour(f"{'English':>10}", "bold", "yellow")
        + colour(f"{'Hindi':>10}", "bold", "yellow")
        + colour(f"{'Marathi':>10}", "bold", "yellow")
    )
    print_rule("─", "muted", 72)

    final_metrics = [
        ("Raw characters", "raw_characters"),
        ("Sentences", "sentences"),
        ("Tokens including punctuation", "tokens_including_punctuation"),
        ("Tokens before stop-word removal", "tokens_before_stopword_removal"),
        ("Unique tokens before removal", "unique_tokens_before_stopword_removal"),
        ("Tokens after stop-word removal", "tokens_after_stopword_removal"),
        ("Stop words removed", "stopwords_removed"),
        ("Unique tokens after removal", "unique_tokens_after_stopword_removal"),
    ]
    for label, key in final_metrics:
        vals = [str(stats[language][key]) for language in LANGUAGES]
        print(
            colour(f"{label:<42}", "muted")
            + "".join(colour(f"{value:>10}", "green") for value in vals)
        )

    ttr_values = []
    for language in LANGUAGES:
        total = stats[language]["tokens_before_stopword_removal"]
        unique = stats[language]["unique_tokens_before_stopword_removal"]
        ttr_values.append(f"{(unique / total):.3f}" if total else "0.000")
    print(
        colour(f"{'Type-token ratio (before removal)':<42}", "muted")
        + "".join(colour(f"{value:>10}", "green") for value in ttr_values)
    )
    list_sizes = [str(len(STOP_WORDS[language])) for language in LANGUAGES]
    print(
        colour(f"{'Stop-word list size':<42}", "muted")
        + "".join(colour(f"{value:>10}", "green") for value in list_sizes)
    )
    stem_counts_text = [
        str(stats[language]["stemming_applied_tokens"])
        for language in LANGUAGES
    ]
    print(
        colour(f"{'Tokens stemmed':<42}", "muted")
        + "".join(colour(f"{value:>10}", "green") for value in stem_counts_text)
    )

    print_section("MOST FREQUENT WORDS — FULL DATASET", "blue")
    for language in LANGUAGES:
        print(colour(f"  {language}", "bold", "magenta"))
        for word, count in word_counts[language].most_common(10):
            print(f"    {colour(word, 'green')}: {colour(count, 'yellow')}")

    print_section("OUTPUT FILES GENERATED", "cyan")
    output_paths = (processed_path, frequencies_path, comparison_path, summary_path)
    for number, path in enumerate(output_paths, start=1):
        print(f"  {colour(f'[{number}]', 'yellow')} {colour(path.name, 'green')}")

    # Print the real first seven rows only after the full-dataset summary.
    print_detailed_walkthrough(source_rows, columns, sample_rows=7)

def main() -> None:
    try:
        run()
    except (OSError, zipfile.BadZipFile, ET.ParseError, csv.Error, ValueError, KeyError) as error:
        print(f"\n[ERROR] {error}", file=sys.stderr, flush=True)
        print(
            "Check that NLP_Three_Languages_Dataset.csv or "
            "NLP_Three_Languages_Dataset.xlsx is in the same folder as this script, "
            "and that the dataset has ID, English, Hindi, and Marathi columns.",
            file=sys.stderr,
            flush=True,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
