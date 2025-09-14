import os
import re
import json
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple, Optional, Any

import javalang
import javalang.ast
from javalang.tree import (
    CompilationUnit,
    ClassDeclaration,
    InterfaceDeclaration,
    EnumDeclaration,
    MethodDeclaration,
    ConstructorDeclaration,
    FieldDeclaration,
    VariableDeclarator,
    FormalParameter,
    MemberReference,
    MethodInvocation,
    SuperMethodInvocation,
    ClassCreator,
    ReferenceType,
    BasicType,
    LocalVariableDeclaration,
    MethodReference,
    ArrayCreator,
    Cast,
    BinaryOperation,  # 用于 instanceof
)

logger = logging.getLogger(__name__)

EXCLUDED_QUALIFIERS = {"this", "super"}
JAVA_LANG_IMPLICIT = "java.lang"


@dataclass
class JavaCodeComponent:
    """
    Represents one code component in a Java repo:
    - class / interface / enum (ID: pkg.Outer$Inner)
    - method (ID: pkg.Outer$Inner#method(T1,T2[]))
    - constructor (ID: pkg.Outer$Inner#<init>(T1,T2))
    """
    id: str
    node: Any
    component_type: str  # class|interface|enum|method|constructor
    file_path: str
    relative_path: str
    depends_on: Set[str] = field(default_factory=set)
    source_code: Optional[str] = None
    start_line: int = 0
    end_line: int = 0
    has_docstring: bool = False
    docstring: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "component_type": self.component_type,
            "file_path": self.file_path,
            "relative_path": self.relative_path,
            "depends_on": sorted(self.depends_on),
            "source_code": self.source_code,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "has_docstring": self.has_docstring,
            "docstring": self.docstring,
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "JavaCodeComponent":
        return JavaCodeComponent(
            id=data["id"],
            node=None,
            component_type=data["component_type"],
            file_path=data["file_path"],
            relative_path=data["relative_path"],
            depends_on=set(data.get("depends_on", [])),
            source_code=data.get("source_code"),
            start_line=data.get("start_line", 0),
            end_line=data.get("end_line", 0),
            has_docstring=data.get("has_docstring", False),
            docstring=data.get("docstring", ""),
        )


@dataclass
class ImportContext:
    """
    Stores package/imports for resolving simple names.
    """
    package: str
    regular_imports: Dict[str, str] = field(default_factory=dict)  # simple -> fqcn
    on_demand_imports: List[str] = field(default_factory=list)     # "java.util"
    static_imports: Dict[str, str] = field(default_factory=dict)   # member -> hostTypeFQCN
    static_on_demand_types: List[str] = field(default_factory=list)  # hostTypeFQCN list


class JavaDependencyParser:
    """
    Parses a Java repository and builds a dependency graph between components.
    """

    def __init__(self, repo_path: str):
        self.repo_path = os.path.abspath(repo_path)
        self.components: Dict[str, JavaCodeComponent] = {}
        self.dependency_graph: Dict[str, List[str]] = {}
        self.modules: Set[str] = set()
        self.classes: Set[str] = set()     # all known class/interface/enum IDs (FQCN with $ for inner)
        self._cu_cache: Dict[str, Tuple[List[str], CompilationUnit]] = {}
        self._super_of: Dict[str, Optional[str]] = {}  # classID -> super classID (if resolvable)

    # -------------------- Public API --------------------

    def parse_repository(self) -> Dict[str, JavaCodeComponent]:
        logger.info(f"Parsing Java repository at {self.repo_path}")
        for root, _, files in os.walk(self.repo_path):
            for file in files:
                if not file.endswith(".java"):
                    continue
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(file_path, self.repo_path)
                module_path = self._file_to_module_path(relative_path)
                self.modules.add(module_path)
                self._parse_file(file_path, relative_path, module_path)

        # Resolve dependencies in a second pass
        self._resolve_dependencies()
        # Optionally add "class depends on its methods" edges (excluding constructors by default)
        self._add_class_method_dependencies(exclude_constructor=True)

        logger.info(f"Found {len(self.components)} Java components")
        return self.components

    def save_dependency_graph(self, output_path: str):
        data = {cid: c.to_dict() for cid, c in self.components.items()}
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved Java dependency graph to {output_path}")

    def load_dependency_graph(self, input_path: str) -> Dict[str, JavaCodeComponent]:
        with open(input_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        self.components = {cid: JavaCodeComponent.from_dict(d) for cid, d in raw.items()}
        logger.info(f"Loaded {len(self.components)} Java components from {input_path}")
        return self.components

    # -------------------- Parsing: first pass --------------------

    def _file_to_module_path(self, file_path: str) -> str:
        path = file_path[:-5] if file_path.endswith(".java") else file_path
        return path.replace(os.path.sep, ".")

    def _parse_file(self, file_path: str, relative_path: str, module_path: str):
        try:
            if file_path in self._cu_cache:
                source_lines, cu = self._cu_cache[file_path]
            else:
                with open(file_path, "r", encoding="utf-8") as f:
                    source = f.read()
                source_lines = source.splitlines()
                cu = javalang.parse.parse(source)
                self._cu_cache[file_path] = (source_lines, cu)

            pkg, import_ctx = self._get_package_and_imports(cu)
            # Collect all components (top-level & nested) recursively
            self._collect_components(source_lines, cu, pkg, file_path, relative_path)

            # Record super-class mapping for each type (if resolvable)
            for type_decl in cu.types or []:
                self._record_super_recursive(type_decl, pkg, cu)

        except Exception as e:
            logger.warning(f"Error parsing {file_path}: {e}")

    def _get_package_and_imports(self, cu: CompilationUnit) -> Tuple[str, ImportContext]:
        pkg = cu.package.name if cu.package else ""
        ctx = ImportContext(package=pkg)
        for imp in cu.imports or []:
            path = imp.path  # "java.util.Map" or "com.foo.Bar.BAZ"
            if imp.static:
                if imp.wildcard:
                    # import static com.foo.Bar.*;
                    ctx.static_on_demand_types.append(path)
                else:
                    # import static com.foo.Bar.BAZ;
                    if "." in path:
                        typ, member = path.rsplit(".", 1)
                        ctx.static_imports[member] = typ
            else:
                if imp.wildcard:
                    # import java.util.*;
                    ctx.on_demand_imports.append(path)
                else:
                    # import java.util.Map;
                    simple = path.split(".")[-1]
                    ctx.regular_imports[simple] = path
        return pkg, ctx

    # --------- component collection (recursive, builds classes/components) ---------

    def _collect_components(self, source_lines: List[str], cu: CompilationUnit,
                            pkg: str, file_path: str, relative_path: str):
        for type_decl in cu.types or []:
            self._collect_type_recursive(source_lines, type_decl, pkg, parent_id=None,
                                         file_path=file_path, relative_path=relative_path)

    def _collect_type_recursive(self, source: List[str], type_decl: Any, pkg: str,
                                parent_id: Optional[str], file_path: str, relative_path: str):
        # Compute this type's ID
        if parent_id:
            class_id = f"{parent_id}${type_decl.name}"
        else:
            class_id = f"{pkg}.{type_decl.name}" if pkg else type_decl.name

        # Component type
        if isinstance(type_decl, ClassDeclaration):
            ctype = "class"
        elif isinstance(type_decl, InterfaceDeclaration):
            ctype = "interface"
        elif isinstance(type_decl, EnumDeclaration):
            ctype = "enum"
        else:
            return

        self.classes.add(class_id)

        # Lines and source
        start_line, end_line = self._brace_region(source, getattr(type_decl, "position", None))
        src = self._get_source_segment(source, start_line, end_line)
        jd, has_jd = self._extract_javadoc_above(source, start_line)

        self.components[class_id] = JavaCodeComponent(
            id=class_id, node=type_decl, component_type=ctype,
            file_path=file_path, relative_path=relative_path,
            source_code=src, start_line=start_line, end_line=end_line,
            has_docstring=has_jd, docstring=jd
        )

        # Methods
        for m in getattr(type_decl, "methods", []) or []:
            if isinstance(m, MethodDeclaration):
                m_id = self._method_id(class_id, m)
                m_start, m_end = self._brace_region(source, getattr(m, "position", None))
                m_src = self._get_source_segment(source, m_start, m_end)
                mj, mh = self._extract_javadoc_above(source, m_start)
                self.components[m_id] = JavaCodeComponent(
                    id=m_id, node=m, component_type="method",
                    file_path=file_path, relative_path=relative_path,
                    source_code=m_src, start_line=m_start, end_line=m_end,
                    has_docstring=mh, docstring=mj
                )

        # Constructors
        for cstr in getattr(type_decl, "constructors", []) or []:
            if isinstance(cstr, ConstructorDeclaration):
                c_id = self._method_id(class_id, cstr)
                c_start, c_end = self._brace_region(source, getattr(cstr, "position", None))
                c_src = self._get_source_segment(source, c_start, c_end)
                cj, ch = self._extract_javadoc_above(source, c_start)
                self.components[c_id] = JavaCodeComponent(
                    id=c_id, node=cstr, component_type="constructor",
                    file_path=file_path, relative_path=relative_path,
                    source_code=c_src, start_line=c_start, end_line=c_end,
                    has_docstring=ch, docstring=cj
                )

        # Nested types
        for member in getattr(type_decl, "body", []) or []:
            if isinstance(member, (ClassDeclaration, InterfaceDeclaration, EnumDeclaration)):
                self._collect_type_recursive(source, member, pkg, class_id, file_path, relative_path)

    def _record_super_recursive(self, type_decl: Any, pkg: str, cu: CompilationUnit, parent_id: Optional[str] = None):
        # Determine this type id
        if parent_id:
            class_id = f"{parent_id}${type_decl.name}"
        else:
            pkg_name = cu.package.name if cu.package else pkg
            class_id = f"{pkg_name}.{type_decl.name}" if pkg_name else type_decl.name

        # Resolve super (first extends if multiple)
        super_fqcn = None
        ext = getattr(type_decl, "extends", None)
        if ext:
            first = ext[0] if isinstance(ext, list) else ext
            _, ctx = self._get_package_and_imports(cu)
            super_fqcn = self._resolve_type_to_fqcn(first, pkg, ctx)
        self._super_of[class_id] = super_fqcn

        # Nested recurse
        for member in getattr(type_decl, "body", []) or []:
            if isinstance(member, (ClassDeclaration, InterfaceDeclaration, EnumDeclaration)):
                self._record_super_recursive(member, pkg, cu, class_id)

    # -------------------- Dependency resolving: second pass --------------------

    def _resolve_dependencies(self):
        for comp_id, comp in self.components.items():
            try:
                source, cu = self._cu_cache[comp.file_path]
                pkg, ctx = self._get_package_and_imports(cu)

                if comp.component_type in {"class", "interface", "enum"}:
                    self._collect_class_dependencies(comp, cu, pkg, ctx, source)
                elif comp.component_type in {"method", "constructor"}:
                    self._collect_method_dependencies(comp, cu, pkg, ctx)

                # Keep only in-repo dependencies
                comp.depends_on = {dep for dep in comp.depends_on if self._in_repo(dep)}

            except Exception as e:
                logger.warning(f"[Resolve] Failed deps for {comp_id}: {e}")

    def _in_repo(self, dep: str) -> bool:
        if dep in self.components:
            return True
        if dep in self.classes:
            return True
        if "#" in dep:
            cls = dep.split("#", 1)[0]
            return cls in self.classes
        return False

    # --------- Class-level dependencies (no method body scanning) ---------

    def _collect_class_dependencies(self, comp: JavaCodeComponent, cu: CompilationUnit,
                                    pkg: str, ctx: ImportContext, source_lines: List[str]):
        node = comp.node

        # extends / implements
        ext = getattr(node, "extends", None)
        if ext:
            ext_list = ext if isinstance(ext, list) else [ext]
            for t in ext_list:
                dep = self._resolve_type_to_fqcn(t, pkg, ctx)
                if dep:
                    comp.depends_on.add(dep)

        for t in getattr(node, "implements", []) or []:
            dep = self._resolve_type_to_fqcn(t, pkg, ctx)
            if dep:
                comp.depends_on.add(dep)

        # class annotations
        for ann in getattr(node, "annotations", []) or []:
            dep = self._resolve_type_to_fqcn(getattr(ann, "name", None), pkg, ctx)
            if dep:
                comp.depends_on.add(dep)

        # fields: type + annotations + initializer expression types
        for fd in getattr(node, "fields", []) or []:
            if isinstance(fd, FieldDeclaration):
                dep = self._resolve_type_to_fqcn(fd.type, pkg, ctx)
                if dep:
                    comp.depends_on.add(dep)
                for ann in getattr(fd, "annotations", []) or []:
                    d2 = self._resolve_type_to_fqcn(getattr(ann, "name", None), pkg, ctx)
                    if d2:
                        comp.depends_on.add(d2)
                for d in getattr(fd, "declarators", []) or []:
                    init = getattr(d, "initializer", None)
                    if init is not None:
                        self._walk_expr_types_only(comp, init, pkg, ctx)

        # enum constants' arguments (types in creators/casts)
        if isinstance(node, EnumDeclaration):
            body = getattr(node, "body", None)
            if body and getattr(body, "constants", None):
                for c in body.constants:
                    for arg in getattr(c, "arguments", []) or []:
                        self._walk_expr_types_only(comp, arg, pkg, ctx)

        # method / constructor signatures (return/params/throws/annotations), no bodies
        for m in getattr(node, "methods", []) or []:
            self._collect_signature_types(comp, m, pkg, ctx)
            for ann in getattr(m, "annotations", []) or []:
                d2 = self._resolve_type_to_fqcn(getattr(ann, "name", None), pkg, ctx)
                if d2:
                    comp.depends_on.add(d2)

        for cstr in getattr(node, "constructors", []) or []:
            self._collect_signature_types(comp, cstr, pkg, ctx)
            for ann in getattr(cstr, "annotations", []) or []:
                d2 = self._resolve_type_to_fqcn(getattr(ann, "name", None), pkg, ctx)
                if d2:
                    comp.depends_on.add(d2)

    # --------- Method/Constructor-level dependencies (body scanning) ---------

    def _collect_method_dependencies(self, comp: JavaCodeComponent, cu: CompilationUnit,
                                     pkg: str, ctx: ImportContext):
        node = comp.node
        current_class = comp.id.split("#", 1)[0]

        # Signature: return/params/throws/annotations
        self._collect_signature_types(comp, node, pkg, ctx)
        for ann in getattr(node, "annotations", []) or []:
            dep = self._resolve_type_to_fqcn(getattr(ann, "name", None), pkg, ctx)
            if dep:
                comp.depends_on.add(dep)

        # Param annotations
        for p in getattr(node, "parameters", []) or []:
            for ann in getattr(p, "annotations", []) or []:
                d2 = self._resolve_type_to_fqcn(getattr(ann, "name", None), pkg, ctx)
                if d2:
                    comp.depends_on.add(d2)

        # Local variable names (to avoid treating var.method(...) as Class.method)
        local_vars = self._collect_local_variables(node)

        # Types in body: creators, array creators, casts, reference types, local var types
        self._walk_and_collect_types(comp, node, pkg, ctx, current_class=current_class, local_vars=local_vars)

        # Unqualified method calls: could be static import, otherwise current_class
        for inv in self._iter_nodes_of_type(node, MethodInvocation):
            if inv.member:
                if inv.qualifier is None:
                    # static import?
                    if inv.member in ctx.static_imports:
                        host = ctx.static_imports[inv.member]
                        self._link_call_best_effort(comp, host, inv.member, self._arg_types_of(inv))
                    else:
                        # static on-demand
                        matched = False
                        for host in ctx.static_on_demand_types:
                            self._link_call_best_effort(comp, host, inv.member, self._arg_types_of(inv))
                            matched = True
                        if not matched:
                            # assume current class
                            self._link_call_best_effort(comp, current_class, inv.member, self._arg_types_of(inv))
                else:
                    qual = inv.qualifier
                    if qual not in EXCLUDED_QUALIFIERS and qual not in local_vars:
                        fqn = self._resolve_qualifier_to_class(qual, pkg, ctx)
                        if fqn:
                            self._link_call_best_effort(comp, fqn, inv.member, self._arg_types_of(inv))

        # Qualified static field / method reference like Foo.BAR / Foo.baz() (MemberReference)
        for mr in self._iter_nodes_of_type(node, MemberReference):
            if mr.qualifier and mr.qualifier not in local_vars and mr.qualifier not in EXCLUDED_QUALIFIERS:
                fqn = self._resolve_qualifier_to_class(mr.qualifier, pkg, ctx)
                if fqn:
                    comp.depends_on.add(fqn)

        # Super calls should map to parent class (not current class)
        for sinv in self._iter_nodes_of_type(node, SuperMethodInvocation):
            if sinv.member:
                super_host = self._super_of.get(current_class)
                if super_host:
                    self._link_call_best_effort(comp, super_host, sinv.member, self._arg_types_of(sinv))

        # Method references: ClassName::m / obj::m / ClassName::new
        for mr in self._iter_nodes_of_type(node, MethodReference):
            qual = getattr(mr, "qualifier", None)
            member = getattr(mr, "member", None)
            if qual and member:
                host = self._resolve_qualifier_to_class(self._strip_this_super(qual), pkg, ctx)
                if host:
                    if member == "new":
                        overloads = self._candidate_overloads(host, "<init>")
                        if overloads:
                            comp.depends_on.update(overloads)
                        else:
                            comp.depends_on.add(host)
                    else:
                        self._link_call_best_effort(comp, host, member, arg_types=None)

        # instanceof（在 javalang 中是 BinaryOperation，operator == 'instanceof'）
        for bo in self._iter_nodes_of_type(node, BinaryOperation):
            if getattr(bo, "operator", None) == "instanceof":
                right = getattr(bo, "operandr", None)
                if right is None:
                    right = getattr(bo, "right", None)  # 兼容不同实现
                dep = self._resolve_type_to_fqcn(right, pkg, ctx)
                if dep:
                    comp.depends_on.add(dep)

    # -------------------- Helpers: building edges --------------------

    def _link_call_best_effort(self, comp: JavaCodeComponent, host_class: str, member: str,
                               arg_types: Optional[List[str]]):
        """
        Try to link to the specific overloaded method. If unresolved:
        - link to all overloads with the same name; if none exist
        - fall back to class-level dependency.
        """
        if arg_types:
            sig = f"{host_class}#{member}({','.join(arg_types)})"
            if sig in self.components:
                comp.depends_on.add(sig)
                return

        overloads = self._candidate_overloads(host_class, member)
        if overloads:
            comp.depends_on.update(overloads)
            return

        if host_class in self.classes:
            comp.depends_on.add(host_class)

    def _candidate_overloads(self, host_class: str, member: str) -> List[str]:
        pref = f"{host_class}#{member}("
        return [cid for cid in self.components.keys() if cid.startswith(pref)]

    def _arg_types_of(self, inv: Any) -> List[str]:
        types: List[str] = []
        args = getattr(inv, "arguments", []) or []
        for a in args:
            if isinstance(a, ClassCreator):
                types.append(self._type_to_str(a.type))
            elif isinstance(a, ArrayCreator):
                types.append(self._type_to_str(a.type))
            elif isinstance(a, Cast):
                types.append(self._type_to_str(a.type))
            elif isinstance(a, ReferenceType):
                types.append(self._type_to_str(a))
            else:
                types.append("?")
        return types

    # -------------------- Tree walking & collectors --------------------

    def _collect_signature_types(self, comp: JavaCodeComponent, node: Any, pkg: str, ctx: ImportContext):
        rtype = getattr(node, "return_type", None)
        dep = self._resolve_type_to_fqcn(rtype, pkg, ctx)
        if dep:
            comp.depends_on.add(dep)

        for p in getattr(node, "parameters", []) or []:
            if isinstance(p, FormalParameter):
                dep = self._resolve_type_to_fqcn(p.type, pkg, ctx)
                if dep:
                    comp.depends_on.add(dep)

        for t in getattr(node, "throws", []) or []:
            dep = self._resolve_type_to_fqcn(t, pkg, ctx)
            if dep:
                comp.depends_on.add(dep)

    def _walk_and_collect_types(self, comp: JavaCodeComponent, node: Any, pkg: str, ctx: ImportContext,
                                current_class: str, local_vars: Set[str]):
        for cc in self._iter_nodes_of_type(node, ClassCreator):
            dep = self._resolve_type_to_fqcn(cc.type, pkg, ctx)
            if dep:
                comp.depends_on.add(dep)

        for ac in self._iter_nodes_of_type(node, ArrayCreator):
            dep = self._resolve_type_to_fqcn(ac.type, pkg, ctx)
            if dep:
                comp.depends_on.add(dep)

        for cs in self._iter_nodes_of_type(node, Cast):
            dep = self._resolve_type_to_fqcn(cs.type, pkg, ctx)
            if dep:
                comp.depends_on.add(dep)

        for rt in self._iter_nodes_of_type(node, ReferenceType):
            dep = self._resolve_type_to_fqcn(rt, pkg, ctx)
            if dep:
                comp.depends_on.add(dep)

        for lvd in self._iter_nodes_of_type(node, LocalVariableDeclaration):
            dep = self._resolve_type_to_fqcn(lvd.type, pkg, ctx)
            if dep:
                comp.depends_on.add(dep)

    def _walk_expr_types_only(self, comp: JavaCodeComponent, expr: Any, pkg: str, ctx: ImportContext):
        for cc in self._iter_nodes_of_type(expr, ClassCreator):
            dep = self._resolve_type_to_fqcn(cc.type, pkg, ctx)
            if dep:
                comp.depends_on.add(dep)
        for ac in self._iter_nodes_of_type(expr, ArrayCreator):
            dep = self._resolve_type_to_fqcn(ac.type, pkg, ctx)
            if dep:
                comp.depends_on.add(dep)
        for cs in self._iter_nodes_of_type(expr, Cast):
            dep = self._resolve_type_to_fqcn(cs.type, pkg, ctx)
            if dep:
                comp.depends_on.add(dep)
        for rt in self._iter_nodes_of_type(expr, ReferenceType):
            dep = self._resolve_type_to_fqcn(rt, pkg, ctx)
            if dep:
                comp.depends_on.add(dep)

    def _iter_nodes_of_type(self, node: Any, t):
        for _, n in self._walk(node):
            if isinstance(n, t):
                yield n

    def _walk(self, node: Any):
        if isinstance(node, javalang.ast.Node):
            for attr in node.attrs:
                child = getattr(node, attr)
                if isinstance(child, list):
                    for c in child:
                        yield from self._walk(c)
                else:
                    yield from self._walk(child)
            yield (None, node)
        elif isinstance(node, list):
            for c in node:
                yield from self._walk(c)
        # primitives ignored

    def _collect_local_variables(self, method_node: Any) -> Set[str]:
        names = set()
        for p in getattr(method_node, "parameters", []) or []:
            if isinstance(p, FormalParameter) and getattr(p, "name", None):
                names.add(p.name)
        for lvd in self._iter_nodes_of_type(method_node, LocalVariableDeclaration):
            for d in getattr(lvd, "declarators", []) or []:
                if isinstance(d, VariableDeclarator) and getattr(d, "name", None):
                    names.add(d.name)
        return names

    # -------------------- Name/Type resolution --------------------

    def _ref_full_name(self, t: ReferenceType) -> str:
        parts = [t.name]
        st = getattr(t, "sub_type", None)
        while st is not None:
            parts.append(st.name)
            st = getattr(st, "sub_type", None)
        return ".".join(parts)

    # def _type_to_str(self, t: Any) -> str:
    #     if t is None:
    #         return "void"
    #     if isinstance(t, BasicType):
    #         dims = "[]" * (getattr(t, "dimensions", 0) or 0)
    #         return t.name + dims
    #     if isinstance(t, ReferenceType):
    #         name = self._ref_full_name(t)
    #         dims = "[]" * (getattr(t, "dimensions", 0) or 0)
    #         return name + dims
    #     if isinstance(t, str):
    #         return t
    #     name = getattr(t, "name", None)
    #     return (name or "?")
    
    def _dims_count(self, obj) -> int:
        dim = getattr(obj, "dimensions", 0)
        if isinstance(dim, list):
            return len(dim)
        if isinstance(dim, int):
            return dim
        return 0

    def _type_to_str(self, t: Any, *, varargs: bool = False) -> str:
        if t is None:
            return "void"
        if isinstance(t, BasicType):
            n = self._dims_count(t) + (1 if varargs else 0)
            return t.name + "[]" * n
        if isinstance(t, ReferenceType):
            name = self._ref_full_name(t)
            n = self._dims_count(t) + (1 if varargs else 0)
            return name + "[]" * n
        if isinstance(t, str):
            return t
        # 兜底：一些节点只有 name，没有明确类型类
        name = getattr(t, "name", None) or "?"
        n = self._dims_count(t) + (1 if varargs else 0)
        return name + ("[]" * n if name != "?" else "")

    # def _method_sig(self, node: Any) -> str:
    #     ps = []
    #     for p in getattr(node, "parameters", []) or []:
    #         ps.append(self._type_to_str(getattr(p, "type", None)))
    #     return "(" + ",".join(ps) + ")"

    def _method_sig(self, node: Any) -> str:
        ps = []
        for p in getattr(node, "parameters", []) or []:
            ps.append(self._type_to_str(getattr(p, "type", None), varargs=getattr(p, "varargs", False)))
        return "(" + ",".join(ps) + ")"

    def _method_id(self, class_id: str, node: Any) -> str:
        if isinstance(node, ConstructorDeclaration):
            return f"{class_id}#<init>{self._method_sig(node)}"
        return f"{class_id}#{node.name}{self._method_sig(node)}"

    def _strip_this_super(self, qual: str) -> str:
        return re.sub(r"\.(this|super)$", "", qual)

    def _resolve_type_to_fqcn(self, t: Any, pkg: str, ctx: ImportContext) -> Optional[str]:
        if t is None:
            return None
        if isinstance(t, BasicType):
            return None
        if isinstance(t, ReferenceType):
            name = self._ref_full_name(t)
        elif isinstance(t, str):
            name = t
        else:
            name = getattr(t, "name", None)
        if not name:
            return None

        if name in self.classes:
            return name

        parts = name.split(".")
        simple = parts[0]
        rest = ".".join(parts[1:]) if len(parts) > 1 else ""

        if simple in ctx.regular_imports:
            base = ctx.regular_imports[simple]
            if rest:
                dotted = f"{base}.{rest}"
                if dotted in self.classes:
                    return dotted
                dollar = f"{base}${rest.replace('.', '$')}"
                if dollar in self.classes:
                    return dollar
                return dotted
            return base

        cand = self._fqcn(pkg, name)
        if cand in self.classes:
            return cand
        cand2 = self._fqcn(pkg, simple)
        if cand2 in self.classes:
            return cand2

        if "." in name:
            dollar = name.replace(".", "$")
            cand3 = self._fqcn(pkg, dollar)
            if cand3 in self.classes:
                return cand3

        for prefix in ctx.on_demand_imports:
            dotted = f"{prefix}.{name}"
            if dotted in self.classes:
                return dotted
            dotted_simple = f"{prefix}.{simple}"
            if dotted_simple in self.classes:
                return dotted_simple
            if "." in name:
                dollar = f"{prefix}.{name.replace('.', '$')}"
                if dollar in self.classes:
                    return dollar

        jl = f"{JAVA_LANG_IMPLICIT}.{name}"
        if jl in self.classes:
            return jl

        return name if "." in name else None

    def _resolve_qualifier_to_class(self, qualifier: str, pkg: str, ctx: ImportContext) -> Optional[str]:
        if not qualifier or qualifier in EXCLUDED_QUALIFIERS:
            return None
        qualifier = self._strip_this_super(qualifier)

        if "." in qualifier:
            if qualifier in self.classes:
                return qualifier
            dollar = qualifier.replace(".", "$")
            if dollar in self.classes:
                return dollar

        if qualifier in ctx.regular_imports:
            return ctx.regular_imports[qualifier]

        cand = self._fqcn(pkg, qualifier)
        if cand in self.classes:
            return cand
        dollar2 = cand.replace(".", "$")
        if dollar2 in self.classes:
            return dollar2

        for prefix in ctx.on_demand_imports:
            base = f"{prefix}.{qualifier}"
            if base in self.classes:
                return base
            base_dollar = f"{prefix}.{qualifier.replace('.', '$')}"
            if base_dollar in self.classes:
                return base_dollar

        return None

    def _fqcn(self, package: str, simple: str) -> str:
        return f"{package}.{simple}" if package else simple

    # -------------------- Text utilities --------------------

    def _get_source_segment(self, lines: List[str], start: int, end: int) -> str:
        if start <= 0 or end <= 0 or end < start:
            return ""
        return "\n".join(lines[start - 1:end])

    def _extract_javadoc_above(self, lines: List[str], start_line: int) -> Tuple[str, bool]:
        i = start_line - 2
        while i >= 0 and (lines[i].strip() == "" or lines[i].lstrip().startswith("@")):
            i -= 1
        if i < 0:
            return "", False

        if "/**" in lines[i] and "*/" in lines[i] and lines[i].find("/**") < lines[i].find("*/"):
            return lines[i].strip(), True

        if "*/" not in lines[i]:
            return "", False
        end = i
        j = end
        acc = []
        opened = False
        while j >= 0:
            acc.append(lines[j])
            if "/**" in lines[j]:
                opened = True
                break
            j -= 1
        if not opened:
            return "", False
        return "\n".join(reversed(acc)), True

    def _brace_region(self, lines: List[str], position) -> Tuple[int, int]:
        if position is None or position.line is None:
            return (1, len(lines))
        start_line = position.line

        open_line_idx = None
        for i in range(start_line - 1, len(lines)):
            if "{" in lines[i]:
                open_line_idx = i
                break
        if open_line_idx is not None:
            depth = 0
            for k in range(open_line_idx, len(lines)):
                for ch in lines[k]:
                    if ch == "{":
                        depth += 1
                    elif ch == "}":
                        depth -= 1
                        if depth == 0:
                            return (start_line, k + 1)
            return (start_line, len(lines))

        for i in range(start_line - 1, min(len(lines), start_line + 200)):
            if ";" in lines[i]:
                return (start_line, i + 1)

        return (start_line, min(len(lines), start_line + 50))

    # -------------------- Add class->methods edges --------------------

    def _add_class_method_dependencies(self, exclude_constructor: bool = True):
        class_to_methods: Dict[str, List[str]] = {}
        for cid, c in self.components.items():
            if c.component_type in {"method", "constructor"}:
                cls = cid.split("#", 1)[0]
                name_part = cid.split("#", 1)[1].split("(", 1)[0] if "#" in cid else ""
                if exclude_constructor and name_part == "<init>":
                    continue
                class_to_methods.setdefault(cls, []).append(cid)
        for cls, mids in class_to_methods.items():
            if cls in self.components:
                self.components[cls].depends_on.update(mids)