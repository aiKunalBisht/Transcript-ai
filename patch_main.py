#!/usr/bin/env python3
"""
patch_readme.py
Switches the HF Spaces frontmatter from sdk: streamlit to sdk: docker.
Removes sdk_version / python_version / app_file (irrelevant once HF
builds from the Dockerfile instead of a managed Streamlit runtime).

Run from project root: python patch_readme.py
"""

with open("README.md", "r", encoding="utf-8") as f:
    content = f.read()

old_block = """---
title: TranscriptAI
emoji: \U0001f9e0
colorFrom: pink
colorTo: red
sdk: streamlit
sdk_version: "1.32.0"
python_version: "3.10"
app_file: app.py
pinned: false
---"""

new_block = """---
title: TranscriptAI
emoji: \U0001f9e0
colorFrom: pink
colorTo: red
sdk: docker
pinned: false
---"""

assert old_block in content, (
    "FAILED: exact frontmatter block not found in README.md — "
    "stopped, no changes made. Paste your current head -n 20 README.md "
    "again and the script will be adjusted."
)

content = content.replace(old_block, new_block)

with open("README.md", "w", encoding="utf-8") as f:
    f.write(content)

print("ok  README.md — sdk: streamlit -> sdk: docker")
print("ok  README.md — removed sdk_version, python_version, app_file")
print()
print("Verify with: head -n 10 README.md")