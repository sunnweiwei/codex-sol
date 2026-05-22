"""Custom zero-dependency apply_patch parser, seek_sequence searcher, and target mutation engine.

Provides Lark-parity behavior and verbatim error diagnostics.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional, Tuple, Set, Dict


# =============================================================================
# Custom Exception Definitions
# =============================================================================

class InvalidPatchError(Exception):
    """Raised when boundaries, preamble headers, or block syntax fails parsing."""
    def __init__(self, message: str) -> None:
        super().__init__(message)


class InvalidHunkError(Exception):
    """Raised when a specific hunk block contains illegal formatting."""
    def __init__(self, message: str, line_number: int) -> None:
        super().__init__(f"Line {line_number}: {message}")
        self.message = message
        self.line_number = line_number


# =============================================================================
# AST Node Models / Data Classes
# =============================================================================

@dataclass
class UpdateFileChunk:
    """A single localized replacement segment inside an UpdateFile hunk."""
    change_context: str | None = None
    old_lines: list[str] = field(default_factory=list)
    new_lines: list[str] = field(default_factory=list)
    is_end_of_file: bool = False


@dataclass
class Hunk:
    """Base class representing a patch operation hunk."""
    path: Path

    def resolve_path(self, cwd: Path) -> Path:
        """Resolves the hunk path against a target working directory."""
        resolved = Path(self.path)
        if resolved.is_absolute():
            return resolved
        return Path(os.path.normpath(cwd / resolved))

    def target_path(self) -> Path:
        """Returns the path affected by this hunk, using the move destination for rename hunks."""
        return self.path


@dataclass
class AddFile(Hunk):
    """Represents a file addition operation hunk."""
    contents: str


@dataclass
class DeleteFile(Hunk):
    """Represents a file deletion operation hunk."""
    pass


@dataclass
class UpdateFile(Hunk):
    """Represents a file update operation hunk, with an optional move/rename target."""
    move_path: Path | None = None
    chunks: list[UpdateFileChunk] = field(default_factory=list)

    def target_path(self) -> Path:
        if self.move_path is not None:
            return self.move_path
        return self.path

    def resolve_move_path(self, cwd: Path) -> Path | None:
        if self.move_path is None:
            return None
        resolved = Path(self.move_path)
        if resolved.is_absolute():
            return resolved
        return Path(os.path.normpath(cwd / resolved))


@dataclass
class ApplyPatchArgs:
    """Unified container representing a successfully parsed patch payload."""
    hunks: list[Hunk]
    patch: str
    workdir: str | None = None
    environment_id: str | None = None


# =============================================================================
# Custom Regex-Based Recursive Descent Parser
# =============================================================================

# Define all strict syntax markers
BEGIN_PATCH_MARKER = "*** Begin Patch"
ENVIRONMENT_ID_MARKER = "*** Environment ID: "
END_PATCH_MARKER = "*** End Patch"
ADD_FILE_MARKER = "*** Add File: "
DELETE_FILE_MARKER = "*** Delete File: "
UPDATE_FILE_MARKER = "*** Update File: "
MOVE_TO_MARKER = "*** Move to: "
EOF_MARKER = "*** End of File"
CHANGE_CONTEXT_MARKER = "@@ "
EMPTY_CHANGE_CONTEXT_MARKER = "@@"


def parse_patch(patch_text: str) -> ApplyPatchArgs:
    """Parses a patch text payload under lenient heredoc extraction rules.
    
    Port of the upstream parse_patch engine (defaulting to lenient mode).
    """
    # 1. Trim the outer patch payload and collect split lines
    trimmed_patch = patch_text.strip()
    original_lines = trimmed_patch.splitlines()

    # 2. Assert boundaries under strict / lenient modes
    patch_lines, hunk_lines = _check_patch_boundaries_lenient(original_lines)

    # 3. Process preamble (Environment ID)
    environment_id, remaining_lines, line_number = _parse_environment_id_preamble(hunk_lines)

    # 4. Iteratively parse subsequent hunks
    hunks: list[Hunk] = []
    lines_slice = remaining_lines
    while lines_slice:
        hunk, parsed_count = _parse_one_hunk(lines_slice, line_number)
        hunks.append(hunk)
        line_number += parsed_count
        lines_slice = lines_slice[parsed_count:]

    # 5. Extract target patch text (excluding heredoc wrappers)
    final_patch_body = "\n".join(patch_lines)

    return ApplyPatchArgs(
        hunks=hunks,
        patch=final_patch_body,
        workdir=None,
        environment_id=environment_id
    )


def _check_start_and_end_lines_strict(first_line: str | None, last_line: str | None) -> None:
    first_trimmed = first_line.strip() if first_line is not None else None
    last_trimmed = last_line.strip() if last_line is not None else None

    if first_trimmed == BEGIN_PATCH_MARKER and last_trimmed == END_PATCH_MARKER:
        return
    
    if first_trimmed != BEGIN_PATCH_MARKER:
        raise InvalidPatchError("The first line of the patch must be '*** Begin Patch'")
    
    raise InvalidPatchError("The last line of the patch must be '*** End Patch'")


def _check_patch_boundaries_strict(lines: list[str]) -> tuple[list[str], list[str]]:
    if not lines:
        first, last = None, None
    elif len(lines) == 1:
        first, last = lines[0], lines[0]
    else:
        first, last = lines[0], lines[-1]

    _check_start_and_end_lines_strict(first, last)
    return lines, lines[1:-1]


def _check_patch_boundaries_lenient(original_lines: list[str]) -> tuple[list[str], list[str]]:
    # Try strict matching first
    try:
        return _check_patch_boundaries_strict(original_lines)
    except InvalidPatchError as strict_err:
        original_parse_error = strict_err

    # Check for lenient heredoc wrappers (must have at least 4 lines total)
    if len(original_lines) >= 4:
        first, last = original_lines[0], original_lines[-1]
        
        is_heredoc_start = first in ("<<EOF", "<<'EOF'", "<<\"EOF\"")
        is_heredoc_end = last.endswith("EOF")

        if is_heredoc_start and is_heredoc_end:
            inner_lines = original_lines[1:-1]
            return _check_patch_boundaries_strict(inner_lines)
        
    raise original_parse_error


def _parse_environment_id_preamble(hunk_lines: list[str]) -> tuple[str | None, list[str], int]:
    if not hunk_lines:
        return None, hunk_lines, 2

    first_line = hunk_lines[0]
    first_trimmed = first_line.lstrip()
    if not first_trimmed.startswith(ENVIRONMENT_ID_MARKER):
        return None, hunk_lines, 2

    # Extract ID and trim spaces
    environment_id = first_trimmed[len(ENVIRONMENT_ID_MARKER):].strip()
    if not environment_id:
        raise InvalidPatchError("apply_patch environment_id cannot be empty")

    return environment_id, hunk_lines[1:], 3


def _parse_one_hunk(lines: list[str], line_number: int) -> tuple[Hunk, int]:
    first_line = lines[0].strip()

    # Case 1: Add File Hunk
    if first_line.startswith(ADD_FILE_MARKER):
        target_path = Path(first_line[len(ADD_FILE_MARKER):])
        contents_lines: list[str] = []
        parsed_lines = 1

        for add_line in lines[1:]:
            if add_line.startswith('+'):
                contents_lines.append(add_line[1:])
                parsed_lines += 1
            else:
                break

        # Compiles added file contents, adding trailing newline to each line
        contents = "".join(l + "\n" for l in contents_lines)
        return AddFile(path=target_path, contents=contents), parsed_lines

    # Case 2: Delete File Hunk
    if first_line.startswith(DELETE_FILE_MARKER):
        target_path = Path(first_line[len(DELETE_FILE_MARKER):])
        return DeleteFile(path=target_path), 1

    # Case 3: Update File Hunk
    if first_line.startswith(UPDATE_FILE_MARKER):
        target_path = Path(first_line[len(UPDATE_FILE_MARKER):])
        remaining_lines = lines[1:]
        parsed_lines = 1

        move_path: Path | None = None
        if remaining_lines and remaining_lines[0].startswith(MOVE_TO_MARKER):
            move_path = Path(remaining_lines[0][len(MOVE_TO_MARKER):])
            remaining_lines = remaining_lines[1:]
            parsed_lines += 1

        chunks: list[UpdateFileChunk] = []
        while remaining_lines:
            first_remaining = remaining_lines[0]
            
            # Skip empty prefix lines between chunks/hunks
            if not first_remaining.strip():
                parsed_lines += 1
                remaining_lines = remaining_lines[1:]
                continue

            # Check if we hit the next hunk header block
            if first_remaining.startswith('*'):
                break

            chunk, chunk_lines_count = _parse_update_file_chunk(
                remaining_lines,
                line_number + parsed_lines,
                allow_missing_context=(len(chunks) == 0)
            )
            chunks.append(chunk)
            parsed_lines += chunk_lines_count
            remaining_lines = remaining_lines[chunk_lines_count:]

        if not chunks:
            raise InvalidHunkError(
                message=f"Update file hunk for path '{target_path}' is empty",
                line_number=line_number
            )

        return UpdateFile(path=target_path, move_path=move_path, chunks=chunks), parsed_lines

    # Raising default error on unmatched line classifications
    raise InvalidHunkError(
        message=(
            f"'{first_line}' is not a valid hunk header. Valid hunk headers: "
            f"'*** Add File: {{path}}', '*** Delete File: {{path}}', '*** Update File: {{path}}'"
        ),
        line_number=line_number
    )


def _parse_update_file_chunk(
    lines: list[str],
    line_number: int,
    allow_missing_context: bool
) -> tuple[UpdateFileChunk, int]:
    if not lines:
        raise InvalidHunkError("Update hunk does not contain any lines", line_number)

    # 1. Parse Context Header
    first_line = lines[0]
    
    if first_line == EMPTY_CHANGE_CONTEXT_MARKER:
        change_context = None
        start_index = 1
    elif first_line.startswith(CHANGE_CONTEXT_MARKER):
        change_context = first_line[len(CHANGE_CONTEXT_MARKER):]
        start_index = 1
    else:
        if not allow_missing_context:
            raise InvalidHunkError(
                message=f"Expected update hunk to start with a @@ context marker, got: '{first_line}'",
                line_number=line_number
            )
        change_context = None
        start_index = 0

    if start_index >= len(lines):
        raise InvalidHunkError("Update hunk does not contain any lines", line_number + 1)

    chunk = UpdateFileChunk(
        change_context=change_context,
        old_lines=[],
        new_lines=[],
        is_end_of_file=False
    )

    # 2. Iterate Gutter Prefix Lines
    parsed_lines = 0
    for line in lines[start_index:]:
        if line == EOF_MARKER:
            if parsed_lines == 0:
                raise InvalidHunkError("Update hunk does not contain any lines", line_number + 1)
            chunk.is_end_of_file = True
            parsed_lines += 1
            break
        
        if not line:
            # Empty line
            chunk.old_lines.append("")
            chunk.new_lines.append("")
        elif line.startswith(' '):
            # Context line
            content = line[1:]
            chunk.old_lines.append(content)
            chunk.new_lines.append(content)
        elif line.startswith('+'):
            # Addition line
            chunk.new_lines.append(line[1:])
        elif line.startswith('-'):
            # Removal line
            chunk.old_lines.append(line[1:])
        else:
            # Encountered terminal non-gutter line
            if parsed_lines == 0:
                raise InvalidHunkError(
                    message=(
                        f"Unexpected line found in update hunk: '{line}'. Every line should start with "
                        f"' ' (context line), '+' (added line), or '-' (removed line)"
                    ),
                    line_number=line_number + 1
                )
            # Break out, leaving line to be parsed by outer state machines
            break
        
        parsed_lines += 1

    return chunk, parsed_lines + start_index


# =============================================================================
# Context Matching Strategy (seek_sequence)
# =============================================================================

def seek_sequence(
    lines: list[str],
    pattern: list[str],
    start: int,
    eof: bool
) -> int | None:
    """Finds the sequence pattern lines within candidate lines starting at or after start index.
    
    Implements character-for-character parity with upstream seek_sequence logic.
    """
    if not pattern:
        return start

    if len(pattern) > len(lines):
        return None

    # Determine starting scanning index
    search_start = len(lines) - len(pattern) if (eof and len(lines) >= len(pattern)) else start

    # Maximum checkable range
    max_search_limit = len(lines) - len(pattern)

    # Pass 1: Exact verification
    for i in range(search_start, max_search_limit + 1):
        if lines[i : i + len(pattern)] == pattern:
            return i

    # Pass 2: Right-trim trailing whitespace verification
    for i in range(search_start, max_search_limit + 1):
        matched = True
        for p_idx, pat in enumerate(pattern):
            if lines[i + p_idx].rstrip('\r\n\t ') != pat.rstrip('\r\n\t '):
                matched = False
                break
        if matched:
            return i

    # Pass 3: Trim trailing & leading whitespace verification
    for i in range(search_start, max_search_limit + 1):
        matched = True
        for p_idx, pat in enumerate(pattern):
            if lines[i + p_idx].strip() != pat.strip():
                matched = False
                break
        if matched:
            return i

    # Pass 4: Unicode normalisation verification
    for i in range(search_start, max_search_limit + 1):
        matched = True
        for p_idx, pat in enumerate(pattern):
            if _normalise_unicode_string(lines[i + p_idx]) != _normalise_unicode_string(pat):
                matched = False
                break
        if matched:
            return i

    return None


def _normalise_unicode_string(s: str) -> str:
    """Normalises Typographical spaces, dashes, hyphens, and single/double quotes to ASCII equivalents."""
    trimmed = s.strip()
    normalized_chars: list[str] = []
    
    for c in trimmed:
        # Fancy Dashes / Hyphens mapping
        if c in ('\u2010', '\u2011', '\u2012', '\u2013', '\u2014', '\u2015', '\u2212'):
            normalized_chars.append('-')
        # Fancy Single Quotes mapping
        elif c in ('\u2018', '\u2019', '\u201a', '\u201b'):
            normalized_chars.append("'")
        # Fancy Double Quotes mapping
        elif c in ('\u201c', '\u201d', '\u201e', '\u201f'):
            normalized_chars.append('"')
        # Typographical Odd/Non-breaking Spaces mapping
        elif c in (
            '\u00a0', '\u2002', '\u2003', '\u2004', '\u2005', '\u2006', 
            '\u2007', '\u2008', '\u2009', '\u200a', '\u202f', '\u205f', '\u3000'
        ):
            normalized_chars.append(' ')
        else:
            normalized_chars.append(c)
            
    return "".join(normalized_chars)


# =============================================================================
# Mutator Application Engine
# =============================================================================

def apply_patch_to_file(file_content: str, hunks: list[Hunk], path: Path, cwd: Path) -> str:
    """Applies a sequence of parsed hunks to active file content under strict mutation state boundaries."""
    original_lines = file_content.split('\n')
    
    # Trim final empty element representing a trailing newline (popped to prevent offset indexing shift)
    has_trailing_newline = False
    if original_lines and original_lines[-1] == "":
        original_lines.pop()
        has_trailing_newline = True

    # Build replacements schedules: list of tuples (insert_index, remove_count, new_lines_list)
    replacements: list[tuple[int, int, list[str]]] = []
    line_index = 0

    for hunk in hunks:
        if not isinstance(hunk, UpdateFile):
            continue
            
        for chunk in hunk.chunks:
            # 1. Adjust running line pointer based on change_context
            if chunk.change_context is not None:
                match_idx = seek_sequence(
                    original_lines,
                    [chunk.change_context],
                    line_index,
                    eof=False
                )
                if match_idx is not None:
                    line_index = match_idx + 1
                else:
                    raise Exception(
                        f"Failed to find context '{chunk.change_context}' in {str(path)}"
                    )

            # 2. Case A: Pure addition mutations
            if not chunk.old_lines:
                insertion_idx = len(original_lines) - 1 if (original_lines and original_lines[-1] == "") else len(original_lines)
                replacements.append((insertion_idx, 0, chunk.new_lines))
                continue

            # 3. Case B: Context replacements mutations
            pattern = list(chunk.old_lines)
            match_idx = seek_sequence(original_lines, pattern, line_index, eof=chunk.is_end_of_file)
            
            # Retry mechanism stripping terminating blank lines
            if match_idx is None and pattern and pattern[-1] == "":
                pattern = pattern[:-1]
                match_idx = seek_sequence(original_lines, pattern, line_index, eof=chunk.is_end_of_file)

            if match_idx is not None:
                replacements.append((match_idx, len(pattern), chunk.new_lines))
                line_index = match_idx + len(pattern)
            else:
                expected_lines_str = "\n".join(chunk.old_lines)
                raise Exception(
                    f"Failed to find expected lines in {str(path)}:\n{expected_lines_str}"
                )

    # Apply compiled replacements in reverse order to keep index alignments accurate
    # Sort replacements descending by source target start index
    replacements.sort(key=lambda item: item[0], reverse=True)
    
    modified_lines = list(original_lines)
    for idx, remove_count, new_lines in replacements:
        modified_lines[idx : idx + remove_count] = new_lines

    # Join results back into strings, always enforcing a trailing newline
    final_contents = "\n".join(modified_lines)
    if not final_contents.endswith('\n'):
        final_contents += '\n'
        
    return final_contents


# =============================================================================
# High-Level Dispatch Engine & Execution Core
# =============================================================================

def apply_patch_execution(patch_text: str, cwd: Path) -> str:
    """Executes full patch parsing, disk validation, path remapping, mutations, and file moves.
    
    Port of the upstream apply_patch runner. Returns success reporting details block.
    """
    # 1. Parse active patch
    args = parse_patch(patch_text)
    
    added: list[str] = []
    modified: list[str] = []
    deleted: list[str] = []
    
    # Process each hunk sequentially.
    # Prior changes are persistently committed (no transactional rollback) to match parity!
    for hunk in args.hunks:
        resolved_path = hunk.resolve_path(cwd)
        
        # Case 1: Add File Hunk
        if isinstance(hunk, AddFile):
            # Write new file content, overwriting target, creating parent directories
            resolved_path.parent.mkdir(parents=True, exist_ok=True)
            with open(resolved_path, 'w', encoding='utf-8') as f:
                f.write(hunk.contents)
            
            # Preserve path spelling from hunk
            path_spelling = str(hunk.path)
            added.append(path_spelling)
            
        # Case 2: Delete File Hunk
        elif isinstance(hunk, DeleteFile):
            if not resolved_path.exists():
                raise Exception(f"Failed to delete file {resolved_path}")
            if resolved_path.is_dir():
                raise Exception(f"Failed to delete file {resolved_path}: path is a directory")
                
            resolved_path.unlink()
            
            # Preserve path spelling from hunk
            path_spelling = str(hunk.path)
            deleted.append(path_spelling)
            
        # Case 3: Update File Hunk
        elif isinstance(hunk, UpdateFile):
            if not resolved_path.exists():
                raise Exception(f"Failed to read file to update {resolved_path}: No such file or directory")
            if resolved_path.is_dir():
                raise Exception(f"Failed to read file to update {resolved_path}: Is a directory")
                
            with open(resolved_path, 'r', encoding='utf-8') as f:
                original_contents = f.read()
                
            new_contents = apply_patch_to_file(original_contents, [hunk], resolved_path, cwd)
            
            # Target for updates changes: move path resolves under effective CWD if defined
            write_path = resolved_path
            move_dest = hunk.resolve_move_path(cwd)
            if move_dest is not None:
                write_path = move_dest
                # Create destination parent dirs if needed
                write_path.parent.mkdir(parents=True, exist_ok=True)
                # Overwrite existing destination target, delete source file
                if resolved_path.exists() and resolved_path != write_path:
                    resolved_path.unlink()
                    
            with open(write_path, 'w', encoding='utf-8') as f:
                f.write(new_contents)
                
            # Log as modified using spelling from target hunk (move destination target if moved!)
            path_spelling = str(hunk.target_path())
            modified.append(path_spelling)

    # Compile stdout report to mirror cargo test expected formats
    if not added and not modified and not deleted:
        raise Exception("No files were modified.")
        
    stdout_report_lines = ["Success. Updated the following files:"]
    for path_str in added:
        stdout_report_lines.append(f"A {path_str}")
    for path_str in modified:
        stdout_report_lines.append(f"M {path_str}")
    for path_str in deleted:
        stdout_report_lines.append(f"D {path_str}")
        
    return "\n".join(stdout_report_lines) + "\n"
