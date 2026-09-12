"""
Makes the project root importable from tests/ and demos/.

pytest inserts the directory containing the topmost conftest.py into sys.path,
so `import token_db` resolves from anywhere in the tree.
"""
