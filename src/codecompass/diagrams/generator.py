"""
Diagram Generator for CodeCompass
Disclaimer: LLM Generated

Generates Mermaid diagrams from codebase analysis:
- Architecture diagrams (module/layer relationships)
- Dependency diagrams (import graphs)
- Flow diagrams (entrypoints and call flows)

Usage:
    codecompass diagram . --type architecture --output diagram.md
"""

import ast
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class ModuleInfo:
    """Information about a Python module"""
    name: str
    path: Path
    imports: List[str] = field(default_factory=list)
    classes: List[str] = field(default_factory=list)
    functions: List[str] = field(default_factory=list)
    entrypoints: List[str] = field(default_factory=list)
    layer: Optional[str] = None  # api, service, data, util, etc.


@dataclass 
class ClassInfo:
    """Information about a class"""
    name: str
    module: str
    methods: List[str] = field(default_factory=list)
    bases: List[str] = field(default_factory=list)
    docstring: Optional[str] = None


@dataclass
class DiagramResult:
    """Result of diagram generation"""
    mermaid_code: str
    title: str
    description: str
    modules_analyzed: int
    

# Layer detection patterns
LAYER_PATTERNS = {
    "api": ["api", "route", "endpoint", "view", "controller", "handler", "rest"],
    "service": ["service", "business", "logic", "usecase", "interactor"],
    "data": ["model", "schema", "entity", "repository", "dao", "database", "db"],
    "util": ["util", "helper", "common", "shared", "lib", "tools"],
    "config": ["config", "settings", "constants", "env"],
    "cli": ["cli", "command", "main", "__main__"],
    "test": ["test", "spec", "fixture"],
}


def detect_layer(module_path: Path, module_name: str) -> str:
    """Detect which architectural layer a module belongs to"""
    path_str = str(module_path).lower()
    name_lower = module_name.lower()
    
    for layer, patterns in LAYER_PATTERNS.items():
        for pattern in patterns:
            if pattern in path_str or pattern in name_lower:
                return layer
    
    return "core"  # Default layer


def parse_python_file(file_path: Path) -> Optional[ModuleInfo]:
    """Parse a Python file and extract module information"""
    try:
        content = file_path.read_text(encoding='utf-8')
        tree = ast.parse(content)
    except (SyntaxError, UnicodeDecodeError):
        return None
    
    module_name = file_path.stem
    if module_name == "__init__":
        module_name = file_path.parent.name
    
    info = ModuleInfo(
        name=module_name,
        path=file_path,
    )
    
    # Extract imports
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                info.imports.append(alias.name.split('.')[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                info.imports.append(node.module.split('.')[0])
    
    # Extract classes and functions
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.ClassDef):
            info.classes.append(node.name)
        elif isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
            info.functions.append(node.name)
            # Detect entrypoints
            if _is_entrypoint(node, content):
                info.entrypoints.append(node.name)
    
    # Detect layer
    info.layer = detect_layer(file_path, module_name)
    
    # Remove duplicates
    info.imports = list(set(info.imports))
    
    return info


def _is_entrypoint(node: ast.FunctionDef, content: str) -> bool:
    """Detect if a function is an entrypoint (CLI, API route, etc.)"""
    # Check for common decorators
    entrypoint_decorators = [
        "app.route", "app.get", "app.post", "app.put", "app.delete",
        "router.get", "router.post", "router.put", "router.delete",
        "click.command", "typer.command", "app.command",
        "main", "cli",
    ]
    
    for decorator in node.decorator_list:
        decorator_str = ast.unparse(decorator) if hasattr(ast, 'unparse') else ""
        for pattern in entrypoint_decorators:
            if pattern in decorator_str.lower():
                return True
    
    # Check for main function
    if node.name in ["main", "cli", "run", "start"]:
        return True
    
    return False


def analyze_repository(repo_path: Path) -> Dict[str, ModuleInfo]:
    """Analyze all Python files in a repository"""
    modules = {}
    
    # Find all Python files
    python_files = list(repo_path.rglob("*.py"))
    
    # Filter out common exclusions
    exclusions = ["venv", ".venv", "node_modules", "__pycache__", ".git", "build", "dist"]
    python_files = [
        f for f in python_files 
        if not any(exc in str(f) for exc in exclusions)
    ]
    
    for file_path in python_files:
        info = parse_python_file(file_path)
        if info:
            # Use relative path as key
            try:
                rel_path = file_path.relative_to(repo_path)
                key = str(rel_path).replace("/", ".").replace("\\", ".").rstrip(".py")
            except ValueError:
                key = info.name
            modules[key] = info
    
    return modules


def build_dependency_graph(modules: Dict[str, ModuleInfo], repo_name: str) -> Dict[str, Set[str]]:
    """Build a graph of internal dependencies between modules"""
    graph = defaultdict(set)
    
    # Get all module names for matching
    module_names = set()
    for key, info in modules.items():
        module_names.add(info.name)
        module_names.add(key)
        # Add package names
        parts = key.split(".")
        for i in range(len(parts)):
            module_names.add(".".join(parts[:i+1]))
    
    for key, info in modules.items():
        for imp in info.imports:
            # Check if this is an internal import
            if imp in module_names or imp == repo_name:
                # Find the actual module
                for target_key, target_info in modules.items():
                    if target_info.name == imp or target_key.endswith(imp):
                        if target_key != key:  # No self-loops
                            graph[key].add(target_key)
                        break
    
    return graph


def generate_architecture_diagram(
    repo_path: Path,
    max_modules: int = 30,
    show_functions: bool = False,
) -> DiagramResult:
    """
    Generate an architecture diagram showing module layers and relationships.
    
    Args:
        repo_path: Path to the repository
        max_modules: Maximum modules to include (for readability)
        show_functions: Whether to show function names in modules
    
    Returns:
        DiagramResult with Mermaid code
    """
    repo_path = Path(repo_path).resolve()
    repo_name = repo_path.name
    
    modules = analyze_repository(repo_path)
    
    if not modules:
        return DiagramResult(
            mermaid_code="graph TD\n    A[No Python modules found]",
            title="Architecture Diagram",
            description="No Python modules were found in the repository.",
            modules_analyzed=0,
        )
    
    # Group by layer
    layers: Dict[str, List[ModuleInfo]] = defaultdict(list)
    for key, info in modules.items():
        layers[info.layer].append(info)
    
    # Limit modules per layer for readability
    max_per_layer = max(3, max_modules // len(layers)) if layers else max_modules
    
    # Build Mermaid diagram
    lines = ["graph TD"]
    
    # Define subgraphs for each layer
    layer_order = ["cli", "api", "service", "core", "data", "util", "config"]
    layer_titles = {
        "cli": "CLI / Entry Points",
        "api": "API Layer",
        "service": "Business Logic",
        "core": "Core",
        "data": "Data Layer",
        "util": "Utilities",
        "config": "Configuration",
        "test": "Tests",
    }
    
    node_ids = {}
    node_counter = 0
    
    for layer in layer_order:
        if layer not in layers or layer == "test":
            continue
        
        layer_modules = layers[layer][:max_per_layer]
        
        lines.append(f"    subgraph {layer_titles.get(layer, layer.title())}")
        
        for info in layer_modules:
            node_id = f"M{node_counter}"
            node_ids[info.name] = node_id
            node_counter += 1
            
            # Build node label
            if show_functions and info.functions:
                funcs = info.functions[:3]
                func_str = "<br/>".join(f"• {f}()" for f in funcs)
                if len(info.functions) > 3:
                    func_str += f"<br/>... +{len(info.functions)-3} more"
                lines.append(f"        {node_id}[\"{info.name}<br/>{func_str}\"]")
            else:
                # Show class count if available
                extras = []
                if info.classes:
                    extras.append(f"{len(info.classes)} classes")
                if info.functions:
                    extras.append(f"{len(info.functions)} funcs")
                if extras:
                    lines.append(f"        {node_id}[\"{info.name}<br/><small>{', '.join(extras)}</small>\"]")
                else:
                    lines.append(f"        {node_id}[\"{info.name}\"]")
        
        lines.append("    end")
    
    # Add dependencies
    dep_graph = build_dependency_graph(modules, repo_name)
    
    added_edges = set()
    for source, targets in dep_graph.items():
        source_info = modules.get(source)
        if not source_info or source_info.name not in node_ids:
            continue
        
        source_id = node_ids[source_info.name]
        
        for target in targets:
            target_info = modules.get(target)
            if not target_info or target_info.name not in node_ids:
                continue
            
            target_id = node_ids[target_info.name]
            edge_key = (source_id, target_id)
            
            if edge_key not in added_edges and source_id != target_id:
                lines.append(f"    {source_id} --> {target_id}")
                added_edges.add(edge_key)
    
    # Add styling
    lines.extend([
        "",
        "    %% Styling",
        "    classDef api fill:#e1f5fe,stroke:#01579b",
        "    classDef service fill:#f3e5f5,stroke:#4a148c", 
        "    classDef data fill:#e8f5e9,stroke:#1b5e20",
        "    classDef util fill:#fff3e0,stroke:#e65100",
        "    classDef cli fill:#fce4ec,stroke:#880e4f",
    ])
    
    mermaid_code = "\n".join(lines)
    
    return DiagramResult(
        mermaid_code=mermaid_code,
        title=f"Architecture Diagram: {repo_name}",
        description=f"Shows {len(modules)} modules organized by architectural layer.",
        modules_analyzed=len(modules),
    )


def generate_dependency_diagram(
    repo_path: Path,
    focus_module: Optional[str] = None,
    max_depth: int = 2,
) -> DiagramResult:
    """
    Generate a dependency diagram showing import relationships.
    
    Args:
        repo_path: Path to the repository
        focus_module: Optional module to focus on (show its dependencies)
        max_depth: Maximum depth of dependency traversal
    
    Returns:
        DiagramResult with Mermaid code
    """
    repo_path = Path(repo_path).resolve()
    repo_name = repo_path.name
    
    modules = analyze_repository(repo_path)
    dep_graph = build_dependency_graph(modules, repo_name)
    
    if not modules:
        return DiagramResult(
            mermaid_code="graph LR\n    A[No modules found]",
            title="Dependency Diagram",
            description="No Python modules found.",
            modules_analyzed=0,
        )
    
    lines = ["graph LR"]
    
    # Generate node IDs
    node_ids = {key: f"D{i}" for i, key in enumerate(modules.keys())}
    
    # Add nodes
    for key, info in modules.items():
        node_id = node_ids[key]
        import_count = len(info.imports)
        lines.append(f"    {node_id}[\"{info.name}<br/><small>{import_count} imports</small>\"]")
    
    # Add edges
    added_edges = set()
    for source, targets in dep_graph.items():
        if source not in node_ids:
            continue
        source_id = node_ids[source]
        
        for target in targets:
            if target not in node_ids:
                continue
            target_id = node_ids[target]
            
            edge_key = (source_id, target_id)
            if edge_key not in added_edges:
                lines.append(f"    {source_id} --> {target_id}")
                added_edges.add(edge_key)
    
    mermaid_code = "\n".join(lines)
    
    return DiagramResult(
        mermaid_code=mermaid_code,
        title=f"Dependency Diagram: {repo_name}",
        description=f"Import relationships between {len(modules)} modules.",
        modules_analyzed=len(modules),
    )


def generate_flow_diagram(
    repo_path: Path,
    include_tests: bool = False,
) -> DiagramResult:
    """
    Generate a flow diagram showing entrypoints and their paths.
    
    Args:
        repo_path: Path to the repository
        include_tests: Whether to include test files
    
    Returns:
        DiagramResult with Mermaid code
    """
    repo_path = Path(repo_path).resolve()
    repo_name = repo_path.name
    
    modules = analyze_repository(repo_path)
    
    # Filter for modules with entrypoints
    entrypoint_modules = {
        k: v for k, v in modules.items() 
        if v.entrypoints and (include_tests or v.layer != "test")
    }
    
    if not entrypoint_modules:
        return DiagramResult(
            mermaid_code="graph TD\n    A[No entrypoints detected]",
            title="Flow Diagram",
            description="No entrypoints (CLI commands, API routes) were detected.",
            modules_analyzed=len(modules),
        )
    
    lines = ["graph TD"]
    lines.append("    User((User))")
    
    node_counter = 0
    
    # Group entrypoints by type
    api_entries = []
    cli_entries = []
    other_entries = []
    
    for key, info in entrypoint_modules.items():
        for entry in info.entrypoints:
            entry_data = (info.name, entry, info.layer)
            if info.layer == "api":
                api_entries.append(entry_data)
            elif info.layer == "cli":
                cli_entries.append(entry_data)
            else:
                other_entries.append(entry_data)
    
    # Add CLI entrypoints
    if cli_entries:
        lines.append("    subgraph CLI Commands")
        for module, func, _ in cli_entries[:10]:
            node_id = f"E{node_counter}"
            lines.append(f"        {node_id}[\"{module}.{func}()\"]")
            lines.append(f"    User --> {node_id}")
            node_counter += 1
        lines.append("    end")
    
    # Add API entrypoints
    if api_entries:
        lines.append("    subgraph API Endpoints")
        for module, func, _ in api_entries[:15]:
            node_id = f"E{node_counter}"
            lines.append(f"        {node_id}[\"{func}()\"]")
            lines.append(f"    User --> {node_id}")
            node_counter += 1
        lines.append("    end")
    
    # Add other entrypoints
    if other_entries:
        lines.append("    subgraph Other Entry Points")
        for module, func, _ in other_entries[:5]:
            node_id = f"E{node_counter}"
            lines.append(f"        {node_id}[\"{module}.{func}()\"]")
            lines.append(f"    User --> {node_id}")
            node_counter += 1
        lines.append("    end")
    
    # Styling
    lines.extend([
        "",
        "    %% Styling",
        "    classDef entrypoint fill:#c8e6c9,stroke:#2e7d32",
        "    style User fill:#bbdefb,stroke:#1976d2",
    ])
    
    mermaid_code = "\n".join(lines)
    
    total_entrypoints = len(api_entries) + len(cli_entries) + len(other_entries)
    
    return DiagramResult(
        mermaid_code=mermaid_code,
        title=f"Entrypoints Diagram: {repo_name}",
        description=f"Found {total_entrypoints} entrypoints (CLI commands, API routes, etc.)",
        modules_analyzed=len(modules),
    )


def generate_class_diagram(
    repo_path: Path,
    module_filter: Optional[str] = None,
    max_classes: int = 20,
) -> DiagramResult:
    """
    Generate a class diagram showing class hierarchies.
    
    Args:
        repo_path: Path to the repository
        module_filter: Optional filter to only include classes from specific module
        max_classes: Maximum number of classes to show
    
    Returns:
        DiagramResult with Mermaid code
    """
    repo_path = Path(repo_path).resolve()
    repo_name = repo_path.name
    
    classes: List[ClassInfo] = []
    
    # Find all Python files
    python_files = list(repo_path.rglob("*.py"))
    exclusions = ["venv", ".venv", "node_modules", "__pycache__", ".git"]
    python_files = [f for f in python_files if not any(exc in str(f) for exc in exclusions)]
    
    for file_path in python_files:
        try:
            content = file_path.read_text(encoding='utf-8')
            tree = ast.parse(content)
        except (SyntaxError, UnicodeDecodeError):
            continue
        
        module_name = file_path.stem
        
        if module_filter and module_filter.lower() not in module_name.lower():
            continue
        
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.ClassDef):
                # Get base classes
                bases = []
                for base in node.bases:
                    if isinstance(base, ast.Name):
                        bases.append(base.id)
                    elif isinstance(base, ast.Attribute):
                        bases.append(base.attr)
                
                # Get methods
                methods = []
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        methods.append(item.name)
                
                # Get docstring
                docstring = ast.get_docstring(node)
                
                classes.append(ClassInfo(
                    name=node.name,
                    module=module_name,
                    methods=methods[:5],  # Limit methods shown
                    bases=bases,
                    docstring=docstring[:100] if docstring else None,
                ))
    
    if not classes:
        return DiagramResult(
            mermaid_code="classDiagram\n    class NoClassesFound",
            title="Class Diagram",
            description="No classes found in the repository.",
            modules_analyzed=0,
        )
    
    # Limit classes
    classes = classes[:max_classes]
    
    lines = ["classDiagram"]
    
    # Add classes
    for cls in classes:
        lines.append(f"    class {cls.name} {{")
        for method in cls.methods:
            prefix = "+" if not method.startswith("_") else "-"
            lines.append(f"        {prefix}{method}()")
        lines.append("    }")
    
    # Add inheritance relationships
    class_names = {cls.name for cls in classes}
    for cls in classes:
        for base in cls.bases:
            if base in class_names:
                lines.append(f"    {base} <|-- {cls.name}")
    
    mermaid_code = "\n".join(lines)
    
    return DiagramResult(
        mermaid_code=mermaid_code,
        title=f"Class Diagram: {repo_name}",
        description=f"Shows {len(classes)} classes with their methods and inheritance.",
        modules_analyzed=len(classes),
    )


def save_diagram(result: DiagramResult, output_path: Path, format: str = "md") -> Path:
    """
    Save a diagram to a file.
    
    Args:
        result: DiagramResult to save
        output_path: Path to output file
        format: Output format ('md' for markdown, 'mermaid' for raw)
    
    Returns:
        Path to the saved file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    
    if format == "md":
        content = f"""# {result.title}

{result.description}

Modules analyzed: {result.modules_analyzed}

```mermaid
{result.mermaid_code}
```

---
*Generated by CodeCompass*
"""
    else:
        content = result.mermaid_code
    
    output_path.write_text(content)
    return output_path.resolve()