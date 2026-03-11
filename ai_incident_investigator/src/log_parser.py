"""Log parser module for extracting structured information from raw log text."""

import re
from typing import Any


class LogParser:
    """Parses raw log strings into structured incident data.

    Extracts error types, severity, keywords, source modules, stack traces,
    referenced files, and event timelines from raw log output. Supports both
    Python and Java stack trace formats and common timestamp patterns.

    All regex patterns are compiled at class level for performance.
    """

    # ---------------------------------------------------------------------------
    # Compiled regex patterns (class-level for reuse across instances/calls)
    # ---------------------------------------------------------------------------

    # Exception / error-type patterns
    _RE_EXCEPTION_COLON = re.compile(
        r"(?:^|[\s\(])([A-Z][A-Za-z0-9]*(?:[A-Z][a-z0-9]+)+(?:Error|Exception|Fault|Failure|Timeout))\s*:",
        re.MULTILINE,
    )
    _RE_RAISE = re.compile(
        r"raise\s+([A-Z][A-Za-z0-9]*(?:Error|Exception|Fault|Failure|Timeout))",
        re.MULTILINE,
    )
    _RE_TRACEBACK_LAST_LINE = re.compile(
        r"^([A-Z][A-Za-z0-9]*(?:[A-Z][a-z0-9]+)*(?:Error|Exception|Fault|Failure|Timeout))(?:\s*:\s*(.*))?$",
        re.MULTILINE,
    )
    _RE_JAVA_EXCEPTION = re.compile(
        r"(?:^|:\s+)([a-z][a-z0-9]*(?:\.[a-z][a-z0-9]*)*\.([A-Z][A-Za-z0-9]*(?:Error|Exception|Fault|Failure|Timeout)))",
        re.MULTILINE,
    )
    _RE_GENERIC_EXCEPTION = re.compile(
        r"\b([A-Z][A-Za-z0-9]*(?:Error|Exception))\b",
    )

    # Severity keyword sets (lowercased for matching)
    _HIGH_PATTERNS = re.compile(
        r"timeout|oom\b|out\s*of\s*memory|outofmemory|connection\s+refused"
        r"|retries?\s+exhausted|retry\s+exhausted|deadlock|crash(?:ed)?|fatal",
        re.IGNORECASE,
    )
    _MEDIUM_PATTERNS = re.compile(
        r"validation\s+error|missing\s+config(?:uration)?|deprecat(?:ed|ion)"
        r"|permission\s+denied|not\s+found|unauthorized|forbidden|404|403|401",
        re.IGNORECASE,
    )
    _LOW_PATTERNS = re.compile(
        r"\bwarning\b|\bwarn\b|info[- ]level|deprecation\s+notice|notice\b",
        re.IGNORECASE,
    )

    # Keyword extraction helpers
    _RE_WORD_TOKEN = re.compile(r"[a-z][a-z0-9_]{2,}", re.IGNORECASE)
    _STOP_WORDS: set[str] = {
        "the", "and", "for", "that", "this", "with", "from", "are", "was",
        "were", "been", "being", "have", "has", "had", "does", "did", "but",
        "not", "you", "all", "can", "her", "his", "its", "our", "they",
        "will", "each", "which", "their", "then", "them", "than", "into",
        "could", "would", "should", "about", "after", "before", "more",
        "other", "some", "such", "only", "also", "most", "very", "just",
        "because", "when", "where", "while", "what", "how", "who", "whom",
        "there", "here", "both", "same", "during", "over", "under",
        "again", "once", "further", "any", "between", "through", "above",
        "below", "own", "too", "these", "those", "nor", "get", "got",
        "line", "file", "none", "true", "false", "def", "class", "self",
        "return", "import", "try", "except", "raise", "pass", "print",
    }

    # Source module patterns
    _RE_PY_FILE = re.compile(r'File\s+"([^"]+\.py)"', re.MULTILINE)
    _RE_PY_MODULE = re.compile(r"\b(?:in|from|module)\s+([a-z_][a-z0-9_]*(?:\.[a-z_][a-z0-9_]*)*)", re.IGNORECASE)
    _RE_JAVA_MODULE = re.compile(r"at\s+([a-z][a-z0-9]*(?:\.[a-z][a-z0-9]*)+)\.[A-Z]", re.MULTILINE)

    # Stack trace extraction
    _RE_PYTHON_TRACEBACK = re.compile(
        r"(Traceback \(most recent call last\):.*?)(?=^\S|\Z)",
        re.DOTALL | re.MULTILINE,
    )
    _RE_JAVA_STACKTRACE = re.compile(
        r"((?:Exception|Error|Throwable|Caused by:).*?(?:\n\s+at\s+.+)+)",
        re.DOTALL,
    )

    # Referenced files
    _RE_PY_FILE_LINE = re.compile(r'File\s+"([^"]+)",\s*line\s+(\d+)', re.MULTILINE)
    _RE_JAVA_FILE_LINE = re.compile(r"at\s+[\w.]+\((\w+\.java):(\d+)\)", re.MULTILINE)

    # Timestamps
    _RE_TIMESTAMP_LINE = re.compile(
        r"^[\[]*"
        r"(\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?)"
        r"[\]]*"
        r"[^\S\n]*(.*)",
        re.MULTILINE,
    )
    _RE_TIME_ONLY_LINE = re.compile(
        r"^[\[]*"
        r"(\d{2}:\d{2}:\d{2}(?:\.\d+)?)"
        r"[\]]*"
        r"[^\S\n]*(.*)",
        re.MULTILINE,
    )

    # ---------------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------------

    def parse(self, raw_log: str) -> dict[str, Any]:
        """Parse a raw log string into a structured dictionary.

        Args:
            raw_log: The raw log text to parse.

        Returns:
            A dict with keys: error_type, severity, keywords, source_module,
            stack_trace, summary, referenced_files, timeline.
        """
        error_type = self._extract_error_type(raw_log)
        stack_trace = self._extract_stack_trace(raw_log)
        first_error_line = self._first_error_message_line(raw_log)

        return {
            "error_type": error_type,
            "severity": self._determine_severity(raw_log),
            "keywords": self._extract_keywords(raw_log),
            "source_module": self._extract_source_module(raw_log),
            "stack_trace": stack_trace,
            "summary": self._build_summary(error_type, first_error_line),
            "referenced_files": self._extract_referenced_files(raw_log),
            "timeline": self._extract_timeline(raw_log),
        }

    # ---------------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------------

    def _extract_error_type(self, raw_log: str) -> str:
        """Extract the exception/error class name from the log.

        Args:
            raw_log: The raw log text.

        Returns:
            The exception class name, or ``"Unknown"`` if none is found.
        """
        # Python traceback final exception line (most specific)
        match = self._RE_TRACEBACK_LAST_LINE.search(raw_log)
        if match:
            return match.group(1)

        # Explicit raise statement
        match = self._RE_RAISE.search(raw_log)
        if match:
            return match.group(1)

        # ExceptionName: message
        match = self._RE_EXCEPTION_COLON.search(raw_log)
        if match:
            return match.group(1)

        # Java fully-qualified exception
        match = self._RE_JAVA_EXCEPTION.search(raw_log)
        if match:
            return match.group(2)

        # Generic fallback
        match = self._RE_GENERIC_EXCEPTION.search(raw_log)
        if match:
            return match.group(1)

        return "Unknown"

    def _determine_severity(self, raw_log: str) -> str:
        """Classify log severity as High, Medium, or Low.

        Args:
            raw_log: The raw log text.

        Returns:
            ``"High"``, ``"Medium"``, or ``"Low"``.
        """
        if self._HIGH_PATTERNS.search(raw_log):
            return "High"
        if self._MEDIUM_PATTERNS.search(raw_log):
            return "Medium"
        if self._LOW_PATTERNS.search(raw_log):
            return "Low"

        # Fall back based on error type presence
        if self._RE_GENERIC_EXCEPTION.search(raw_log):
            return "Medium"

        return "Low"

    def _extract_keywords(self, raw_log: str) -> list[str]:
        """Extract meaningful keywords from the log text.

        Args:
            raw_log: The raw log text.

        Returns:
            A deduplicated list of lowercase keyword strings.
        """
        # Focus on error/message lines rather than full stack frames
        error_lines = self._error_relevant_lines(raw_log)
        tokens = self._RE_WORD_TOKEN.findall(error_lines)
        seen: set[str] = set()
        keywords: list[str] = []
        for tok in tokens:
            lower = tok.lower()
            if lower not in self._STOP_WORDS and lower not in seen and not lower.isdigit():
                seen.add(lower)
                keywords.append(lower)
        return keywords

    def _extract_source_module(self, raw_log: str) -> str:
        """Identify the originating source module from the log.

        Args:
            raw_log: The raw log text.

        Returns:
            Module name string, or empty string if not identifiable.
        """
        # Python File "xxx.py" – take the last (deepest) frame
        py_files = self._RE_PY_FILE.findall(raw_log)
        if py_files:
            filename = py_files[-1].replace("\\", "/").rsplit("/", 1)[-1]
            return filename.removesuffix(".py")

        # Java-style at com.package.module.Class
        match = self._RE_JAVA_MODULE.search(raw_log)
        if match:
            fq = match.group(1)
            return fq.rsplit(".", 1)[-1]

        # Explicit module references (e.g. "in event_producer")
        match = self._RE_PY_MODULE.search(raw_log)
        if match:
            return match.group(1).rsplit(".", 1)[-1]

        return ""

    def _extract_stack_trace(self, raw_log: str) -> str:
        """Extract the full stack trace if present.

        Args:
            raw_log: The raw log text.

        Returns:
            The stack trace string, or empty string.
        """
        match = self._RE_PYTHON_TRACEBACK.search(raw_log)
        if match:
            trace = match.group(0).strip()
            # Append the final exception line that follows the traceback
            end = match.end()
            rest = raw_log[end:].lstrip("\n")
            for line in rest.splitlines():
                stripped = line.strip()
                if stripped:
                    trace += "\n" + stripped
                    # Keep going for multi-line exception messages
                    if self._RE_TRACEBACK_LAST_LINE.match(stripped):
                        break
                else:
                    break
            return trace.strip()

        match = self._RE_JAVA_STACKTRACE.search(raw_log)
        if match:
            return match.group(0).strip()

        return ""

    def _extract_referenced_files(self, raw_log: str) -> list[dict[str, str | int]]:
        """Extract file references with line numbers from the log.

        Args:
            raw_log: The raw log text.

        Returns:
            List of ``{"file": str, "line": int}`` dicts.
        """
        seen: set[tuple[str, int]] = set()
        results: list[dict[str, str | int]] = []

        for filepath, lineno in self._RE_PY_FILE_LINE.findall(raw_log):
            key = (filepath, int(lineno))
            if key not in seen:
                seen.add(key)
                results.append({"file": filepath, "line": int(lineno)})

        for filepath, lineno in self._RE_JAVA_FILE_LINE.findall(raw_log):
            key = (filepath, int(lineno))
            if key not in seen:
                seen.add(key)
                results.append({"file": filepath, "line": int(lineno)})

        return results

    def _extract_timeline(self, raw_log: str) -> list[dict[str, str]]:
        """Extract chronologically sorted timestamped events.

        Supports ``YYYY-MM-DDTHH:MM:SS``, ``YYYY-MM-DD HH:MM:SS``,
        ``[YYYY-MM-DD HH:MM:SS]``, and ``HH:MM:SS`` formats.

        Args:
            raw_log: The raw log text.

        Returns:
            Sorted list of ``{"timestamp": str, "event": str}`` dicts.
        """
        events: list[dict[str, str]] = []
        seen_positions: set[int] = set()

        for match in self._RE_TIMESTAMP_LINE.finditer(raw_log):
            seen_positions.add(match.start())
            ts = match.group(1).strip()
            event = match.group(2).strip()
            if event:
                events.append({"timestamp": ts, "event": event})

        for match in self._RE_TIME_ONLY_LINE.finditer(raw_log):
            if match.start() not in seen_positions:
                ts = match.group(1).strip()
                event = match.group(2).strip()
                if event:
                    events.append({"timestamp": ts, "event": event})

        # Sort chronologically (lexicographic works for ISO-like timestamps)
        events.sort(key=lambda e: e["timestamp"])
        return events

    def _first_error_message_line(self, raw_log: str) -> str:
        """Return the first line that looks like an error/exception message.

        Args:
            raw_log: The raw log text.

        Returns:
            The first error line, or the first non-empty line as fallback.
        """
        for line in raw_log.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if re.search(r"(error|exception|fail|fatal|crash|timeout)", stripped, re.IGNORECASE):
                return stripped
        # Fallback: first non-empty line
        for line in raw_log.splitlines():
            stripped = line.strip()
            if stripped:
                return stripped
        return ""

    def _build_summary(self, error_type: str, first_error_line: str) -> str:
        """Construct a one-line failure summary.

        Args:
            error_type: The extracted error/exception class name.
            first_error_line: The first relevant error message line.

        Returns:
            A concise summary string.
        """
        if error_type != "Unknown" and first_error_line:
            # Avoid duplication if the error line already starts with the type
            if first_error_line.startswith(error_type):
                return first_error_line
            return f"{error_type}: {first_error_line}"
        if error_type != "Unknown":
            return error_type
        if first_error_line:
            return first_error_line
        return "Unable to determine failure cause"

    def _error_relevant_lines(self, raw_log: str) -> str:
        """Return lines most likely to contain meaningful error keywords.

        Filters out generic stack-frame lines (``File "...", line N``) and
        focuses on exception messages, logged errors, and summary lines.

        Args:
            raw_log: The raw log text.

        Returns:
            A newline-joined string of relevant lines.
        """
        relevant: list[str] = []
        for line in raw_log.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            # Skip pure stack-frame lines
            if stripped.startswith("File ") and ", line " in stripped:
                continue
            if stripped.startswith("at ") and "(" in stripped and ")" in stripped:
                continue
            if stripped == "Traceback (most recent call last):":
                continue
            relevant.append(stripped)
        return "\n".join(relevant)
