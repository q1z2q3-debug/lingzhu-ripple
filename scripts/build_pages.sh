#!/usr/bin/env bash
# 构建 Pages 入口：把根目录的一页式方案页同步到 docs/index.html
# 用法：bash scripts/build_pages.sh    （Pages 源：main 分支 /docs）
set -euo pipefail
cd "$(dirname "$0")/.."
src="灵助落地一页.html"
dst="docs/index.html"
cp "$src" "$dst"
echo "synced: $src -> $dst ($(wc -c < "$dst") bytes)"
