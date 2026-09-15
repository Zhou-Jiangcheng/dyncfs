"""Check documented signatures and CLI flags using AST, without imports."""
import ast
import copy
import json
from pathlib import Path
import re

DIRECTIVE = re.compile(
    r"^```\{py:(function|method|class)\}\s+([\w.]+)\((.*)\)\s*$",
    re.MULTILINE,
)


def clean_arguments(args, drop_self=False):
    args = copy.deepcopy(args)
    positional = args.posonlyargs + args.args
    if drop_self and positional and positional[0].arg == "self":
        if args.posonlyargs:
            args.posonlyargs.pop(0)
        else:
            args.args.pop(0)
    for arg in args.posonlyargs + args.args + args.kwonlyargs:
        arg.annotation = None
    for arg in (args.vararg, args.kwarg):
        if arg is not None:
            arg.annotation = None
    return ast.dump(args, include_attributes=False)


def check_reference(root):
    root = Path(root)
    docs = root / "docs"
    inventory = json.loads((docs / "api/public-api.json").read_text(encoding="utf-8"))
    errors = []
    if len(inventory) != len(set(inventory)):
        errors.append("Duplicate names in public-api.json")
    found = {}
    modules = {}
    for page in sorted((docs / "api").glob("*.md")):
        text = page.read_text(encoding="utf-8")
        for kind, name, arguments in DIRECTIVE.findall(text):
            if name in found:
                errors.append(f"Duplicate directive: {name}")
            found[name] = page.name
            parts = name.split(".")
            if len(parts) < 3 or parts[0] != "dyncfs":
                errors.append(f"Unexpected API name: {name}")
                continue
            module = parts[1]
            if module not in modules:
                modules[module] = ast.parse(
                    (root / "dyncfs" / f"{module}.py").read_text(encoding="utf-8-sig")
                )
            node = modules[module]
            try:
                for part in parts[2:]:
                    node = next(n for n in node.body if getattr(n, "name", None) == part)
                if kind == "class":
                    node = next(n for n in node.body if getattr(n, "name", None) == "__init__")
                documented = ast.parse(f"def documented({arguments}): pass").body[0]
                actual_args = clean_arguments(node.args, drop_self=kind in ("method", "class"))
                if actual_args != clean_arguments(documented.args):
                    errors.append(f"Signature mismatch: {name}")
            except (StopIteration, AttributeError, SyntaxError) as exc:
                errors.append(f"Invalid reference {name}: {type(exc).__name__}")
    for name in sorted(set(inventory) - set(found)):
        errors.append(f"Missing API directive: {name}")
    for name in sorted(set(found) - set(inventory)):
        errors.append(f"Unlisted API directive: {name}")

    cli_tree = ast.parse((root / "dyncfs/main.py").read_text(encoding="utf-8-sig"))
    flags = {
        node.args[0].value
        for node in ast.walk(cli_tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        and node.func.attr == "add_argument" and node.args
        and isinstance(node.args[0], ast.Constant)
        and isinstance(node.args[0].value, str)
    }
    cli_page = (docs / "cli.md").read_text(encoding="utf-8")
    for flag in sorted(flags):
        if f"`{flag}`" not in cli_page:
            errors.append(f"CLI flag missing: {flag}")
    if errors:
        raise ValueError("\n".join(errors))
    print(f"API check passed: {len(found)} signatures; {len(flags)} CLI flags.")
    return len(found)


if __name__ == "__main__":
    check_reference(Path(__file__).resolve().parents[2])
