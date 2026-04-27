from .utils import (
    add_to_history,
    build_export_json,
    clean_text,
    detect_language,
    export_filename,
    format_history_label,
    language_display_name,
    parse_uploaded_file,
)
from .evaluator import evaluate
from .logger import get_trends, get_stats, get_recent_entries
from .language_intelligence import get_features, detect_hindi_patterns
from .cache import get_cached, set_cache
