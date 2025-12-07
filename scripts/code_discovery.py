#!/usr/bin/env python3
"""
Code Discovery Script for ModernBERT v3.

Extracts all classes, functions, and dataclasses from v3 implementation files
and generates a comprehensive Markdown document.
"""

import ast
import os
from pathlib import Path
from dataclasses import dataclass


@dataclass
class ClassInfo:
    name: str
    bases: list[str]
    methods: list[str]
    docstring: str | None
    line: int


@dataclass
class FunctionInfo:
    name: str
    args: list[str]
    return_type: str | None
    docstring: str | None
    line: int


@dataclass
class FileInfo:
    path: str
    relative_path: str
    classes: list[ClassInfo]
    functions: list[FunctionInfo]
    constants: list[str]
    imports: list[str]


def extract_file_info(filepath: str, base_path: str) -> FileInfo | None:
    """Extract classes, functions from a Python file."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            source = f.read()
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None

    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        print(f"Syntax error in {filepath}: {e}")
        return None

    classes = []
    functions = []
    constants = []
    imports = []

    for node in ast.walk(tree):
        # Extract imports
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                imports.append(f"{module}.{alias.name}")

    for node in ast.iter_child_nodes(tree):
        # Top-level classes
        if isinstance(node, ast.ClassDef):
            bases = []
            for base in node.bases:
                if isinstance(base, ast.Name):
                    bases.append(base.id)
                elif isinstance(base, ast.Attribute):
                    bases.append(
                        f"{base.value.id if hasattr(base.value, 'id') else '...'}.{base.attr}"
                    )

            methods = []
            for item in node.body:
                if isinstance(item, ast.FunctionDef):
                    methods.append(item.name)

            docstring = ast.get_docstring(node)
            classes.append(
                ClassInfo(
                    name=node.name,
                    bases=bases,
                    methods=methods,
                    docstring=(
                        docstring[:200] + "..." if docstring and len(docstring) > 200 else docstring
                    ),
                    line=node.lineno,
                )
            )

        # Top-level functions
        elif isinstance(node, ast.FunctionDef):
            args = []
            for arg in node.args.args:
                args.append(arg.arg)

            return_type = None
            if node.returns:
                if isinstance(node.returns, ast.Name):
                    return_type = node.returns.id
                elif isinstance(node.returns, ast.Constant):
                    return_type = str(node.returns.value)

            docstring = ast.get_docstring(node)
            functions.append(
                FunctionInfo(
                    name=node.name,
                    args=args,
                    return_type=return_type,
                    docstring=(
                        docstring[:150] + "..." if docstring and len(docstring) > 150 else docstring
                    ),
                    line=node.lineno,
                )
            )

        # Top-level constants (UPPER_CASE assignments)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id.isupper():
                    constants.append(target.id)

    rel_path = os.path.relpath(filepath, base_path)
    return FileInfo(
        path=filepath,
        relative_path=rel_path,
        classes=classes,
        functions=functions,
        constants=constants,
        imports=[],  # Skip imports for brevity
    )


def generate_markdown(files: list[FileInfo], output_path: str):
    """Generate Markdown document from file info."""
    lines = []

    lines.append("# ModernBERT v3 Code Discovery Document")
    lines.append("")
    lines.append("> Auto-generated from source code analysis")
    lines.append("> Generated: December 2025")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Summary
    total_classes = sum(len(f.classes) for f in files)
    total_functions = sum(len(f.functions) for f in files)
    total_methods = sum(sum(len(c.methods) for c in f.classes) for f in files)

    lines.append("## Summary")
    lines.append("")
    lines.append(f"- **Total Files:** {len(files)}")
    lines.append(f"- **Total Classes:** {total_classes}")
    lines.append(f"- **Total Top-Level Functions:** {total_functions}")
    lines.append(f"- **Total Methods:** {total_methods}")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Group by directory
    groups = {}
    for f in files:
        parts = f.relative_path.split(os.sep)
        if len(parts) > 1:
            group = parts[-2]  # Parent folder
        else:
            group = "root"
        if group not in groups:
            groups[group] = []
        groups[group].append(f)

    # Output by group
    for group, group_files in sorted(groups.items()):
        lines.append(f"## {group.upper()}")
        lines.append("")

        for f in sorted(group_files, key=lambda x: x.relative_path):
            filename = os.path.basename(f.relative_path)
            lines.append(f"### `{filename}`")
            lines.append("")
            lines.append(f"**Path:** `{f.relative_path}`")
            lines.append("")

            if f.constants:
                lines.append("**Constants:**")
                for const in f.constants[:10]:
                    lines.append(f"- `{const}`")
                if len(f.constants) > 10:
                    lines.append(f"- ... and {len(f.constants) - 10} more")
                lines.append("")

            if f.classes:
                lines.append("**Classes:**")
                lines.append("")
                for cls in f.classes:
                    bases_str = f"({', '.join(cls.bases)})" if cls.bases else ""
                    lines.append(f"#### `class {cls.name}{bases_str}` (line {cls.line})")
                    if cls.docstring:
                        lines.append(f"> {cls.docstring}")
                    if cls.methods:
                        lines.append("")
                        lines.append("Methods:")
                        for method in cls.methods:
                            lines.append(f"- `{method}()`")
                    lines.append("")
                lines.append("")

            if f.functions:
                lines.append("**Functions:**")
                lines.append("")
                for func in f.functions:
                    args_str = ", ".join(func.args[:5])
                    if len(func.args) > 5:
                        args_str += ", ..."
                    ret = f" -> {func.return_type}" if func.return_type else ""
                    lines.append(f"- `{func.name}({args_str}){ret}` (line {func.line})")
                    if func.docstring:
                        lines.append(f"  > {func.docstring}")
                lines.append("")

            lines.append("---")
            lines.append("")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Generated: {output_path}")


def main():
    base_path = Path("d:/Modeling_studio")

    # Files to analyze
    file_patterns = [
        # Models
        "src/modeling_studio/models/config_v3.py",
        "src/modeling_studio/models/hub_tokens.py",
        "src/modeling_studio/models/hub_initialization_v3.py",
        "src/modeling_studio/models/tokenization_v3.py",
        "src/modeling_studio/models/attention_v3.py",
        "src/modeling_studio/models/ffn_v3.py",
        "src/modeling_studio/models/layers_v3.py",
        "src/modeling_studio/models/lora_v3.py",
        "src/modeling_studio/models/poolers_v3.py",
        "src/modeling_studio/models/pair_encoder_v3.py",
        "src/modeling_studio/models/modernbert_v3.py",
        "src/modeling_studio/models/initialization_v3.py",
        "src/modeling_studio/models/embeddings_v3.py",
        "src/modeling_studio/models/encoder_v3.py",
        "src/modeling_studio/models/heads_v3.py",
        "src/modeling_studio/models/losses_v3.py",
        "src/modeling_studio/models/routing_v3.py",
        "src/modeling_studio/models/registry_v3.py",
        "src/modeling_studio/models/verification_v3.py",
        # Trainers
        "src/modeling_studio/trainers/freezing_v3.py",
        "src/modeling_studio/trainers/zipper_lr_v3.py",
        "src/modeling_studio/trainers/schedulers_v3.py",
        "src/modeling_studio/trainers/gradient_utils_v3.py",
        "src/modeling_studio/trainers/gradient_masking_v3.py",
        "src/modeling_studio/trainers/trainer_v3.py",
        "src/modeling_studio/trainers/lr_groups_v3.py",
        "src/modeling_studio/trainers/lora_v3.py",
        # Data
        "src/modeling_studio/data/collators_v3.py",
        "src/modeling_studio/data/loaders_v3.py",
        "src/modeling_studio/data/extractors_v3.py",
        "src/modeling_studio/data/replay_sampler_v3.py",
        "src/modeling_studio/data/shard_loader_v3.py",
        # Training
        "src/modeling_studio/training/losses_v3.py",
        # Scripts
        "scripts/initialize_v3_from_v2.py",
        "scripts/train_v3_phase0_5.py",
        "scripts/train_v3_phase1.py",
    ]

    files = []
    for pattern in file_patterns:
        filepath = base_path / pattern
        if filepath.exists():
            info = extract_file_info(str(filepath), str(base_path))
            if info:
                files.append(info)
                print(
                    f"Processed: {pattern} ({len(info.classes)} classes, {len(info.functions)} functions)"
                )
        else:
            print(f"NOT FOUND: {pattern}")

    output_path = base_path / "docs" / "v3" / "CODE_DISCOVERY.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    generate_markdown(files, str(output_path))


if __name__ == "__main__":
    main()
