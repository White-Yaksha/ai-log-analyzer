"""Context builder module for constructing structured LLM prompts from parsed logs and code snippets."""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

_DEFAULT_INSTRUCTIONS = """\
Analyze the above failure and provide:
1. Incident Summary — a concise description of what went wrong.
2. Root Cause — the underlying technical reason for the failure.
3. Impacted Module — which module or service is most affected.
4. Failure Path — the sequence of events leading to the failure.
5. Suggested Fix — a concrete recommendation to resolve the issue.
6. Severity — your assessment of operational severity (Critical / High / Medium / Low).
7. Confidence Score — rate your confidence in this analysis from 0.0 to 1.0 \
based on how well the retrieved code explains the observed error."""

_DEFAULT_TEMPLATE = """\
=== INCIDENT ANALYSIS REQUEST ===

Error Information:
{error_info}

{timeline_section}\
{stack_trace_section}\
Relevant Code:
{code_section}

Instructions:
{instructions}
"""


class ContextBuilder:
    """Builds structured LLM prompts from parsed incident data and retrieved code snippets.

    Combines error metadata, an optional event timeline, stack trace, and
    relevant source-code snippets into a single prompt string, applying
    truncation to stay within a configurable character budget so the
    resulting prompt fits inside an LLM context window.
    """

    # Overhead estimate for the fixed template chrome (headers, newlines,
    # instructions) so that truncation budgets are calculated accurately.
    _TEMPLATE_OVERHEAD = 350

    def __init__(
        self,
        max_context_length: int = 3000,
        prompt_template: Optional[str] = None,
    ) -> None:
        """Initialise the context builder.

        Args:
            max_context_length: Approximate maximum character length for the
                combined prompt.  When the assembled prompt exceeds this limit,
                code snippets and the stack trace are truncated (in that order)
                to fit.
            prompt_template: Optional Jinja-style format string.  If ``None``,
                a sensible built-in template is used.  The template may contain
                the placeholders ``{error_info}``, ``{timeline_section}``,
                ``{stack_trace_section}``, ``{code_section}``, and
                ``{instructions}``.
        """
        if max_context_length < 200:
            logger.warning(
                "max_context_length=%d is very small; prompt quality may suffer.",
                max_context_length,
            )
        self.max_context_length: int = max_context_length
        self.prompt_template: str = prompt_template or _DEFAULT_TEMPLATE
        self._instructions: str = _DEFAULT_INSTRUCTIONS
        logger.debug(
            "ContextBuilder initialised (max_context_length=%d)", max_context_length
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_prompt(
        self,
        parsed_log: dict,
        code_snippets: list[dict],
    ) -> str:
        """Construct a structured LLM prompt from parsed log data and code snippets.

        Sections are assembled in order: error information, optional timeline,
        optional stack trace, relevant code, and analysis instructions.  If the
        total length exceeds :attr:`max_context_length`, code snippets are
        trimmed first (keeping boosted/high-priority snippets), followed by the
        stack trace.

        Args:
            parsed_log: Dictionary produced by :class:`LogParser`, expected to
                contain keys such as ``error_type``, ``severity``,
                ``keywords``, ``source_module``, ``summary``,
                ``stack_trace``, and optionally ``timeline``.
            code_snippets: List of snippet dicts as returned by
                :class:`Retriever`.  Each dict may contain ``file_path``,
                ``function_name``, ``line_numbers`` (with ``start``/``end``),
                ``snippet``, and ``boosted``.

        Returns:
            The fully formatted prompt string.
        """
        error_info = self._format_error_info(parsed_log)
        instructions = self._indent(self._instructions, indent=2)

        timeline_section = ""
        timeline = parsed_log.get("timeline")
        if timeline:
            timeline_section = (
                "Incident Timeline:\n"
                + self._format_timeline(timeline)
                + "\n\n"
            )

        stack_trace_raw = parsed_log.get("stack_trace", "")

        # --- budget calculation -------------------------------------------
        fixed_length = (
            len(self.prompt_template)
            + len(error_info)
            + len(timeline_section)
            + len(instructions)
            + self._TEMPLATE_OVERHEAD
        )
        remaining = max(self.max_context_length - fixed_length, 0)

        # Allocate remaining budget: 70 % code, 30 % stack trace
        code_budget = int(remaining * 0.70)
        trace_budget = remaining - code_budget

        code_section = self._format_code_snippets(code_snippets, max_length=code_budget)
        if stack_trace_raw:
            truncated_trace = self._truncate(
                self._indent(stack_trace_raw, indent=2), max_length=trace_budget
            )
            stack_trace_section = "Stack Trace:\n" + truncated_trace + "\n\n"
        else:
            stack_trace_section = ""

        prompt = self.prompt_template.format(
            error_info=error_info,
            timeline_section=timeline_section,
            stack_trace_section=stack_trace_section,
            code_section=code_section,
            instructions=instructions,
        )

        # Final safety net: hard-truncate if still over budget
        if len(prompt) > self.max_context_length:
            logger.warning(
                "Prompt length %d exceeds max_context_length %d after sectional "
                "truncation; applying hard truncation.",
                len(prompt),
                self.max_context_length,
            )
            prompt = self._truncate(prompt, self.max_context_length)

        logger.info("Built prompt (%d chars, limit %d)", len(prompt), self.max_context_length)
        return prompt

    # ------------------------------------------------------------------
    # Formatting helpers
    # ------------------------------------------------------------------

    def _format_timeline(self, timeline: list[dict]) -> str:
        """Format timeline entries as vertically aligned text.

        Args:
            timeline: List of ``{"timestamp": str, "event": str}`` dicts,
                assumed to be in chronological order.

        Returns:
            An indented, newline-separated string of timestamp–event pairs.
        """
        if not timeline:
            return ""

        max_ts_len = max(len(entry.get("timestamp", "")) for entry in timeline)

        lines: list[str] = []
        for entry in timeline:
            ts = entry.get("timestamp", "")
            event = entry.get("event", "")
            lines.append(f"  {ts:<{max_ts_len}}  {event}")
        return "\n".join(lines)

    def _format_code_snippets(
        self,
        snippets: list[dict],
        max_length: int,
    ) -> str:
        """Format code snippets with metadata headers, respecting a character budget.

        Boosted (high-priority) snippets are rendered first to ensure they
        survive truncation.  Each snippet is separated by a visual rule and
        includes file path, function name, and line-number range.

        Args:
            snippets: List of snippet dicts (see :meth:`build_prompt`).
            max_length: Maximum total character length for the returned string.

        Returns:
            Formatted code-snippet block, possibly truncated.
        """
        if not snippets:
            return "  (no relevant code snippets found)\n"

        # Sort: boosted first, then by score descending
        ordered = sorted(
            snippets,
            key=lambda s: (not s.get("boosted", False), -(s.get("score", 0.0))),
        )

        parts: list[str] = []
        current_length = 0

        for snip in ordered:
            block = self._render_single_snippet(snip)
            if current_length + len(block) > max_length and parts:
                # No room for this snippet; stop adding
                parts.append("  ... [additional snippets truncated]\n")
                break
            parts.append(block)
            current_length += len(block)

        result = "\n".join(parts)
        if len(result) > max_length:
            result = self._truncate(result, max_length)
        return result

    def _truncate(self, text: str, max_length: int) -> str:
        """Truncate *text* to at most *max_length* characters.

        If truncation is necessary the text is cut and a
        ``... [truncated]`` suffix is appended.

        Args:
            text: The text to potentially truncate.
            max_length: Maximum allowed character length.

        Returns:
            The original text if it fits, otherwise a truncated version
            with an ellipsis suffix.
        """
        suffix = "... [truncated]"
        if max_length <= 0:
            return suffix
        if len(text) <= max_length:
            return text
        cut = max_length - len(suffix)
        if cut <= 0:
            return suffix[:max_length]
        return text[:cut] + suffix

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _format_error_info(parsed_log: dict) -> str:
        """Render the Error Information section from parsed log fields."""
        keywords = parsed_log.get("keywords", [])
        keyword_str = ", ".join(keywords) if keywords else "(none)"

        lines = [
            f"  Type: {parsed_log.get('error_type', 'Unknown')}",
            f"  Severity: {parsed_log.get('severity', 'Unknown')}",
            f"  Keywords: {keyword_str}",
            f"  Module: {parsed_log.get('source_module', 'unknown')}",
            f"  Summary: {parsed_log.get('summary', 'N/A')}",
        ]
        return "\n".join(lines)

    @staticmethod
    def _render_single_snippet(snippet: dict) -> str:
        """Render one code snippet with its metadata header."""
        file_path = snippet.get("file_path", "unknown")
        func = snippet.get("function_name", "")
        line_nums = snippet.get("line_numbers", {})
        start = line_nums.get("start", "?")
        end = line_nums.get("end", "?")
        code = snippet.get("snippet", "")
        boosted = snippet.get("boosted", False)

        tag = "[HIGH PRIORITY - referenced in stack trace] " if boosted else ""
        func_part = f" :: {func}()" if func else ""
        header = f"  --- {tag}{file_path}{func_part} (lines {start}-{end}) ---"

        indented_code = "\n".join(
            f"  {line}" for line in code.splitlines()
        ) if code else "  (empty snippet)"

        return f"{header}\n{indented_code}\n  ---\n"

    @staticmethod
    def _indent(text: str, indent: int = 2) -> str:
        """Indent every line of *text* by *indent* spaces."""
        prefix = " " * indent
        return "\n".join(prefix + line for line in text.splitlines())
