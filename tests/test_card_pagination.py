"""Tests for src/card/render/pagination.py — layout pagination and split_atom."""

from src.card.render.atoms import RenderAtom, estimate_atom_size
from src.card.render.budget import RenderBudget
from src.card.render.layout import SectionLayout, paginate_layout
from src.card.render.pagination import split_atom, stabilize_markdown_split


def _paginate_body(atoms: list[RenderAtom], budget: RenderBudget) -> list[list[RenderAtom]]:
    pages = paginate_layout(
        SectionLayout(sticky_head=(), status=(), body=tuple(atoms), appendix=()),
        budget,
    )
    return [list(page) for page in pages]


class TestPaginateAtoms:
    """Tests for paginate_atoms function."""

    def test_single_page_within_budget(self) -> None:
        """Atoms that fit in one page → 1 page."""
        atoms = [
            RenderAtom(kind="text", content="short text", splittable=True, node_count=1),
        ]
        for a in atoms:
            a.byte_size = estimate_atom_size(a)

        budget = RenderBudget()
        pages = _paginate_body(atoms, budget)

        assert len(pages) == 1
        assert len(pages[0]) == 1
        assert pages[0][0].content == "short text"

    def test_multi_page_split(self) -> None:
        """Large content → multiple pages."""
        # Create content that exceeds byte budget
        # Budget: 27*1024 - 500 = ~27148 bytes available
        # Each atom with 5000 chars → ~15100 bytes estimated
        # So 2 atoms should overflow to 2 pages
        large_content = "A" * 5000
        atoms = [
            RenderAtom(kind="text", content=large_content, splittable=False, node_count=1),
            RenderAtom(kind="text", content=large_content, splittable=False, node_count=1),
            RenderAtom(kind="text", content=large_content, splittable=False, node_count=1),
        ]
        for a in atoms:
            a.byte_size = estimate_atom_size(a)

        # Use a small budget to force pagination
        budget = RenderBudget(byte_budget=16000)
        pages = _paginate_body(atoms, budget)

        assert len(pages) > 1

    def test_atom_split_by_paragraph(self) -> None:
        """Split on double newline."""
        content = "paragraph one\n\nparagraph two\n\nparagraph three"
        atom = RenderAtom(kind="text", content=content, splittable=True, node_count=1)
        atom.byte_size = estimate_atom_size(atom)

        # Remaining bytes allows only first paragraph
        # "paragraph one" → ~13*3 + 100 = 139 bytes
        # "paragraph one\n\nparagraph two" → ~29*3 + 100 = 187 bytes
        remaining = 150
        result = split_atom(atom, remaining)

        assert result is not None
        assert len(result) == 2
        assert result[0].content == "paragraph one"
        assert "paragraph two" in result[1].content
        assert "paragraph three" in result[1].content

    def test_atom_split_by_line(self) -> None:
        """Split on single newline."""
        content = "line one\nline two\nline three\nline four"
        atom = RenderAtom(kind="text", content=content, splittable=True, node_count=1)
        atom.byte_size = estimate_atom_size(atom)

        # Remaining bytes allows first line only
        remaining = 150
        result = split_atom(atom, remaining)

        assert result is not None
        assert len(result) == 2
        # First part should have at least line one
        assert "line one" in result[0].content
        # Second part should have remaining lines
        assert result[1].content  # non-empty

    def test_atom_split_by_chars(self) -> None:
        """Split at 1600 chars."""
        # Content with no newlines, longer than 1600 chars
        content = "x" * 3200
        atom = RenderAtom(kind="text", content=content, splittable=True, node_count=1)
        atom.byte_size = estimate_atom_size(atom)

        # Small remaining forces char split
        remaining = 200
        result = split_atom(atom, remaining)

        assert result is not None
        assert len(result) == 2
        # Both parts should have content
        assert len(result[0].content) > 0
        assert len(result[1].content) > 0
        # Total content should be preserved
        assert result[0].content + result[1].content == content

    def test_text_split_inside_code_fence_makes_each_part_parseable(self) -> None:
        content = "```go\nline one\nline two\nline three\n```\nafter"
        atom = RenderAtom(kind="text", content=content, splittable=True, node_count=1)
        atom.byte_size = estimate_atom_size(atom)

        result = split_atom(atom, remaining_bytes=150)

        assert result is not None
        assert result[0].content.endswith("\n```")
        assert result[1].content.startswith("```\n")
        assert result[0].content.count("```") % 2 == 0
        assert result[1].content.count("```") % 2 == 0

    def test_long_fences_preserve_delimiter_and_require_matching_closer(self) -> None:
        open_cases = (
            ("````markdown\npayload\n```", "````"),
            ("````markdown\npayload\n~~~~", "````"),
            ("~~~~text\npayload\n~~~", "~~~~"),
            ("~~~~text\npayload\n````", "~~~~"),
            ("````markdown\npayload\n`````not-a-closer", "````"),
        )
        for first_content, fence in open_cases:
            first, rest = stabilize_markdown_split(first_content, "remaining")
            assert first.splitlines()[-1] == fence
            assert rest.splitlines()[0] == fence

        closed_cases = (
            "````markdown\npayload\n`````",
            "~~~~text\npayload\n~~~~~~  ",
        )
        for first_content in closed_cases:
            assert stabilize_markdown_split(first_content, "remaining") == (
                first_content,
                "remaining",
            )

    def test_text_split_inside_inline_code_makes_each_part_parseable(self) -> None:
        content = "`" + ("x" * 3200) + "` after"
        atom = RenderAtom(kind="text", content=content, splittable=True, node_count=1)
        atom.byte_size = estimate_atom_size(atom)

        result = split_atom(atom, remaining_bytes=200)

        assert result is not None
        assert result[0].content.endswith("`")
        assert result[1].content.startswith("`")
        assert result[0].content.count("`") % 2 == 0
        assert result[1].content.count("`") % 2 == 0

    def test_different_inline_backtick_length_does_not_close_span(self) -> None:
        first_content = "``code with ` literal``"

        assert stabilize_markdown_split(first_content, "remaining") == (
            first_content,
            "remaining",
        )

    def test_no_content_lost(self) -> None:
        """All paragraph text is preserved after pagination (split separators at page boundaries are not duplicated)."""
        paragraphs = [f"Paragraph {i} with some content." for i in range(20)]
        content = "\n\n".join(paragraphs)
        atoms = [
            RenderAtom(kind="text", content=content, splittable=True, node_count=1),
        ]
        for a in atoms:
            a.byte_size = estimate_atom_size(a)

        # Small budget to force splitting
        budget = RenderBudget(byte_budget=2000)
        pages = _paginate_body(atoms, budget)

        # Collect all content from all pages
        total_content = ""
        for page in pages:
            for atom in page:
                total_content += atom.content

        # Every paragraph must appear in the combined output
        for p in paragraphs:
            assert p in total_content, f"Missing paragraph: {p}"
        # No content should start with a leading separator (no stray newlines at page start)
        for page in pages:
            for atom in page:
                assert not atom.content.startswith("\n"), f"Atom starts with newline: {atom.content[:40]!r}"

    def test_empty_atoms(self) -> None:
        """Empty list → [[]]."""
        budget = RenderBudget()
        pages = _paginate_body([], budget)

        assert pages == [[]]
