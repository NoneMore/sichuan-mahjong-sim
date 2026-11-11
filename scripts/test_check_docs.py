"""Small regression checks for the dependency-free Markdown checker."""

import tempfile
import unittest
from pathlib import Path

from check_docs import anchor_ids, check_markdown, visible_markdown


class MarkdownCheckTests(unittest.TestCase):
    def test_ignores_fenced_links_and_headings(self):
        shown, errors = visible_markdown(
            '# Real\n\n```md\n# Fake\n[broken](missing.md)\n```\n', Path('test.md')
        )
        self.assertEqual(errors, [])
        self.assertEqual(anchor_ids(shown), {'real'})
        self.assertNotIn('missing.md', shown)

    def test_unclosed_fence(self):
        _, errors = visible_markdown('```rust\nlet x = 1;\n', Path('test.md'))
        self.assertEqual(len(errors), 1)
        self.assertIn('unclosed', errors[0])

    def test_repeated_heading_slugs(self):
        self.assertEqual(anchor_ids('# Same\n## Same\n### API `Example`'),
                         {'same', 'same-1', 'api-example'})

    def test_local_paths_and_anchors(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            (root / 'docs').mkdir()
            index = root / 'README.md'
            target = root / 'docs' / 'guide.md'
            index.write_text('[Valid](docs/guide.md#intro)\n[External](https://example.com)\n', encoding='utf-8')
            target.write_text('# Intro\n', encoding='utf-8')
            self.assertEqual(check_markdown(root, [index, target]), [])
            index.write_text('[Broken](docs/guide.md#wrong)\n[Missing](nothing.md)\n', encoding='utf-8')
            self.assertEqual(len(check_markdown(root, [index, target])), 2)

    def test_directory_link_and_relative_escape(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            file = root / 'README.md'
            (root / 'docs').mkdir()
            file.write_text('[Directory](docs/)\n[Escape](../outside.md)\n', encoding='utf-8')
            errors = check_markdown(root, [file])
            self.assertEqual(len(errors), 1)
            self.assertIn('leaves repository', errors[0])


if __name__ == '__main__':
    unittest.main()
