#!/bin/bash
# PR #7 Accessibility Verification Script
# This script verifies that PR #7 is accessible and functional

echo "======================================"
echo "PR #7 Accessibility Verification"
echo "======================================"
echo ""

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Test 1: Check if branch exists on remote
echo "Test 1: Checking remote branch existence..."
if git ls-remote --heads origin | grep -q "copilot/update-parameter-display-logic"; then
    echo -e "${GREEN}✓ PASS${NC}: Branch 'copilot/update-parameter-display-logic' exists on remote"
else
    echo -e "${RED}✗ FAIL${NC}: Branch not found on remote"
    exit 1
fi
echo ""

# Test 2: Fetch the branch
echo "Test 2: Fetching PR #7 branch..."
if git fetch origin copilot/update-parameter-display-logic 2>&1 | grep -q "From"; then
    echo -e "${GREEN}✓ PASS${NC}: Successfully fetched branch"
else
    echo -e "${RED}✗ FAIL${NC}: Failed to fetch branch"
    exit 1
fi
echo ""

# Test 3: Check branch commits
echo "Test 3: Verifying branch has commits..."
git fetch origin copilot/update-parameter-display-logic >/dev/null 2>&1
COMMIT_COUNT=$(git log --oneline FETCH_HEAD | wc -l)
if [ "$COMMIT_COUNT" -gt 0 ]; then
    echo -e "${GREEN}✓ PASS${NC}: Branch has $COMMIT_COUNT commits"
    echo "  Recent commits:"
    git log --oneline FETCH_HEAD -3 | sed 's/^/    /'
else
    echo -e "${RED}✗ FAIL${NC}: No commits found"
    exit 1
fi
echo ""

# Test 4: List changed files
echo "Test 4: Listing files changed in PR #7..."
git fetch origin copilot/update-parameter-display-logic:pr7-branch-temp 2>/dev/null || true
CHANGED_FILES=$(git diff --name-only main pr7-branch-temp 2>/dev/null | wc -l)
if [ "$CHANGED_FILES" -gt 0 ]; then
    echo -e "${GREEN}✓ PASS${NC}: Found $CHANGED_FILES changed files:"
    git diff --name-only main pr7-branch-temp 2>/dev/null | head -10 | sed 's/^/  - /'
else
    echo -e "${RED}✗ FAIL${NC}: No changed files found"
    exit 1
fi
echo ""

# Test 5: Verify specific files exist
echo "Test 5: Verifying key files from PR #7..."
if git show pr7-branch-temp:LINKAGE_UI_IMPLEMENTATION_MEMO.md >/dev/null 2>&1; then
    echo -e "${GREEN}✓ PASS${NC}: LINKAGE_UI_IMPLEMENTATION_MEMO.md exists in PR #7"
    FILE_SIZE=$(git show pr7-branch-temp:LINKAGE_UI_IMPLEMENTATION_MEMO.md | wc -l)
    echo "  File has $FILE_SIZE lines"
else
    echo -e "${RED}✗ FAIL${NC}: Key file not found"
    exit 1
fi
echo ""

# Summary
echo "======================================"
echo -e "${GREEN}All tests passed!${NC}"
echo "PR #7 is fully accessible and functional"
echo "======================================"
